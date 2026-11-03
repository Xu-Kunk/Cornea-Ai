"""HPA-FT — Head-Proximal-Aware Fine-Tuning of Geneformer V2-104M (this study main method).

Motivation (this study layer-wise diagnostic, locked): naive classification
fine-tuning collapses the epithelial differentiation gradient of the CLS
embedding at the FINAL hidden state (Spearman rho 0.806 pretrained -> ~ -0.04
naive), while the 13-layer measurement localises the collapse: L1-L4 RETAIN
gradient (L3 peaks at 0.8380, above frozen 0.7656) and L5-L12 COLLAPSE (L12
-0.0424). CAFT (lambda=0.5, continuity constrained on the FINAL representation
only, hidden state 12) never executed its continuity term (a dead hook; the archived +0.210
was run-to-run nondeterminism, an honest negative) and never
recovers the collapse; the causal-head-ablation (C2) shows the collapse lives
inside the body weights, i.e. classification training *overwrites* the
head-proximal (upper) layers.

HPA-FT is the direct training-side answer: instead of constraining only the
already-overwritten final representation, it is LAYER-AWARE —

  variant 1  LWC (layer-wise continuity):   the CAFT-style cosine-adjacency
              continuity term is distributed over SPECIFIED intermediate
              hidden states (--cont-layers, default L3-L10) instead of only
              the final one. Each cont layer has its own stop-gradient layer
              centroid matrix, precomputed over the full train set at each
              epoch start (identical machinery to CAFT).
  variant 2  HEADFREEZE (head-region freeze/damp): encoder layers at/above a
              hidden-state threshold (--freeze-upper-from, default 5 = L5-L12)
              are either FROZEN (requires_grad=False, --freeze-upper) or
              DAMPED via per-layer learning-rate groups (--upper-lr-scale<1);
              only the early layers (L1-L4) + head are trained, testing
              directly whether NOT rewriting the head-proximal layers
              preserves the gradient.
  --mode lwc | headfreeze | both  selects the mechanisms; both variants keep
  the CAFT-style final-representation continuity as an OPTION (--final-cont).

Loss (mode-dependent):
    L = L_cls + lambda * mean_{c in cont_layers} L_cont_c
    L_cont_c(z) = -mean(cos(z_CLS^{(c)}, centroid^{(c)}_{layer_adj(x)}))
    with per-layer stop-gradient centroids (identical to CAFT, Methods §4.3).
    Non-epithelial cells (layer == -1) keep L_cls but contribute no
    continuity term, exactly as in CAFT.

Everything else mirrors the locked arms: same tokenizer / donor split /
Classifier.prepare_data / HF training arguments / bf16 weights / group_by_length.
Only the loss placement and the layer freeze/damp policy differ.

NOTE on frozen cont layers: when the upper encoder is frozen, cont layers at/above
the freeze threshold produce CONSTANT hidden states, so their continuity terms
carry no gradient. They are silently dropped from the effective set (and a
warning is logged); only trainable layers receive the constraint.
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812  (mirrors corneal_fm/caft.py style)
from transformers import TrainerCallback, TrainingArguments

from corneal_fm.caft import NON_LAYER, CAFTTrainer, class_to_layer_map

# hidden_states index 0 = embedding output, 1..12 = transformer layers,
# 12 == last_hidden_state (the paper's layer-12 rho). Same indexing as
# scripts/measure_layerwise_rho.py.
N_HIDDEN_STATES = 13
FINAL_HIDDEN = N_HIDDEN_STATES - 1  # 12

# Mode-aware default cont-layer specs (hidden-state indices, as in the
# layer-wise rho diagnostic):
#   lwc        -> L3-L10: the mid/upper-body band straddling the preserved
#                 (L3,L4) and collapsing (L5+) regions; directly answers the
#                 CAFT failure (constraints distributed onto layers where
#                 gradient still flows instead of only the overwritten final).
#   headfreeze -> L2-L4: the trainable early band that feeds the frozen
#                 region (embedding output L0 excluded: too shallow).
#   both       -> same trainable early band as headfreeze (upper frozen).
MODE_DEFAULT_CONT = {"lwc": "3-10", "headfreeze": "2-4", "both": "2-4"}
VALID_MODES = ("lwc", "headfreeze", "both")


def parse_cont_layers(spec: str) -> list[int]:
    """Parse a --cont-layers spec into a sorted list of hidden-state indices.

    Accepts comma-separated ints, inclusive ranges ("3-10"), and the literal
    token "final" (== hidden state 12, the CAFT final representation). All
    values must be in 1..12 (transformer layer hidden states).
    """
    layers: list[int] = []
    for tok in str(spec).lower().split(","):
        tok = tok.strip()
        if not tok:
            continue
        if tok == "final":
            layers.append(FINAL_HIDDEN)
        elif "-" in tok:
            lo_s, hi_s = tok.split("-", 1)
            lo, hi = int(lo_s), int(hi_s)
            if lo > hi:
                lo, hi = hi, lo
            layers.extend(range(lo, hi + 1))
        else:
            layers.append(int(tok))
    out: list[int] = []
    seen: set[int] = set()
    for hid in layers:
        if hid not in seen:
            seen.add(hid)
            out.append(hid)
    for hid in out:
        if not (1 <= hid <= FINAL_HIDDEN):
            raise ValueError(
                f"cont layer must be in 1..{FINAL_HIDDEN} (hidden-state index); got {hid}"
            )
    return sorted(out)


def freeze_upper_layers(model: Any, freeze_from_hidden: int = 5) -> tuple[int, int]:
    """Freeze encoder layers whose hidden-state index >= freeze_from_hidden.

    Mapping (rho-diagnostic indexing): hidden state h is the output of
    encoder.layer[h-1]; the collapsed band L5-L12 corresponds to
    encoder.layer[4..11] (0-indexed). Returns (n_frozen_params, n_trainable_params).
    The head, pooler and embeddings stay trainable.
    """
    n_frozen = 0
    n_encoder = int(model.config.num_hidden_layers)
    first = max(0, min(freeze_from_hidden - 1, n_encoder))
    for i in range(first, n_encoder):
        for p in model.bert.encoder.layer[i].parameters():
            p.requires_grad = False
            n_frozen += p.numel()
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return n_frozen, n_trainable


def count_parameters(model: Any) -> tuple[int, int]:
    """(n_trainable, n_total) parameters (used by feasibility/freeze checks)."""
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    return n_trainable, n_total


class HPATrainer(CAFTTrainer):
    """HF Trainer implementing HPA-FT (LWC + headfreeze, combinable).

    Inherits CAFTTrainer's centroid machinery; overrides the precompute to
    build PER-CONT-LAYER centroid matrices and the loss to apply continuity
    over every effective cont layer. The freeze/damp policy is applied in
    __init__ (before HF creates the optimizer, so frozen params are excluded
    and damped params get their own LR group).

    FIX 2026-08-05 (P0): the inherited epoch-begin precompute hook is dead code
    under transformers 4.57 (``on_epoch_begin`` is dispatched only to
    registered TrainerCallback objects), so the LWC continuity term silently
    never ran. :meth:`_precompute_centroids` is now ragged-safe (per-batch
    padding, epi-only forwarding, bf16-consistent centroids) and is invoked by
    :class:`HPAEpochBeginCallback`, registered in :func:`build_hpa_trainer`
    (mirrors the ordinal.py fix).
    """

    def __init__(
        self,
        hpa_lambda: float,
        class_to_layer: dict,
        mode: str = "both",
        cont_layers: str | list[int] | None = None,
        final_cont: bool = False,
        freeze_upper: bool = False,
        freeze_from_hidden: int = 5,
        upper_lr_scale: float = 1.0,
        n_layers: int = 5,
        token_dictionary_file: str | Path | None = None,
        centroid_batch: int = 512,
        **kwargs,
    ):
        if mode not in VALID_MODES:
            raise ValueError(f"mode must be one of {VALID_MODES}; got {mode!r}")
        self.mode = mode
        self.hpa_lambda = float(hpa_lambda)
        self.final_cont = bool(final_cont)
        self.freeze_from_hidden = int(freeze_from_hidden)
        if not (1 <= self.freeze_from_hidden <= N_HIDDEN_STATES):
            raise ValueError(
                f"freeze_from_hidden must be in 1..{N_HIDDEN_STATES}; got {freeze_from_hidden}"
            )
        self.upper_lr_scale = float(upper_lr_scale)

        # -- freeze/damp policy -------------------------------------------
        if mode in ("headfreeze", "both"):
            # Default: freeze. Damp only when the user explicitly asks for a
            # <1.0 scale WITHOUT --freeze-upper.
            self.freeze_upper = bool(freeze_upper) or self.upper_lr_scale >= 1.0
            self.damp_upper = not self.freeze_upper
        else:  # lwc: no layer freeze/damp (warn below if requested)
            self.freeze_upper = False
            self.damp_upper = False
            if freeze_upper or upper_lr_scale < 1.0:
                print(
                    "[HPA] WARN mode='lwc': --freeze-upper/--upper-lr-scale are "
                    "headfreeze mechanisms and are IGNORED.",
                    flush=True,
                )

        # -- effective cont layers ----------------------------------------
        spec = cont_layers if cont_layers is not None else MODE_DEFAULT_CONT[mode]
        if isinstance(spec, str):
            requested = parse_cont_layers(spec)
        else:
            requested = sorted(int(x) for x in spec)
        if self.final_cont and FINAL_HIDDEN not in requested:
            requested.append(FINAL_HIDDEN)
            requested.sort()
        if self.freeze_upper:
            kept = [hid for hid in requested if hid < self.freeze_from_hidden]
            dropped = [hid for hid in requested if hid >= self.freeze_from_hidden]
            if dropped:
                print(
                    f"[HPA] WARN cont layers {dropped} are at/above the freeze "
                    f"threshold (hidden {self.freeze_from_hidden}); they yield "
                    f"constant hidden states and are dropped. Effective: {kept}",
                    flush=True,
                )
            requested = kept
        self.cont_layers: list[int] = requested

        super().__init__(
            caft_lambda=hpa_lambda,
            class_to_layer=class_to_layer,
            n_layers=n_layers,
            token_dictionary_file=token_dictionary_file,
            centroid_batch=centroid_batch,
            **kwargs,
        )

        self.cont_layer_centroids: torch.Tensor | None = None

        if self.freeze_upper:
            n_frozen, n_train = freeze_upper_layers(self.model, self.freeze_from_hidden)
            print(
                f"[HPA] FROZEN hidden states {self.freeze_from_hidden}..12 "
                f"({n_frozen / 1e6:.2f}M params frozen; {n_train / 1e6:.2f}M trainable)",
                flush=True,
            )

    # -- per-layer centroid precompute (overrides CAFT) --------------------

    def _precompute_centroids(self) -> None:
        """Ragged-safe, per-cont-layer centroid precompute (FIX 2026-08-05, P0).

        Overrides the inherited CAFTTrainer version, which (a) crashes on the
        variable-length geneformer-prepared dataset (``torch.tensor`` on ragged
        ``input_ids`` raises) and (b) was never invoked because transformers
        4.57 dispatches ``on_epoch_begin`` only to registered TrainerCallback
        objects, never to a Trainer method. Here each slice is padded to its
        own max length (right-pad with pad_id, matching the collator), the
        attention mask derives from the ``length`` column via :meth:`_attn_mask`,
        and a detached centroid matrix per effective cont layer is accumulated:
        shape ``(n_cont, n_layers=5, hidden_size)``, L2-normalised over the
        hidden dim. Skipped when lambda==0 or no effective cont layers.

        Cost guard: only EPITHELIAL cells are forwarded (non-epithelial cells
        map to layer -1 and contribute nothing to any centroid, so skipping
        their forwards yields identical centroids at ~1/3 of the cost).
        Centroids are allocated in the model-weight dtype (bf16 on GPU) for
        consistency with the bf16 training weights.
        """
        if self.hpa_lambda == 0 or not self.cont_layers:
            self.cont_layer_centroids = None
            return
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
        n_cont = len(self.cont_layers)
        sums = torch.zeros(
            n_cont,
            self.n_layers,
            self.model.config.hidden_size,
            device=device,
            dtype=dtype,
        )
        counts = torch.zeros(n_cont, self.n_layers, device=device)
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
                out = self.model.bert(
                    input_ids=ids,
                    attention_mask=self._attn_mask(ids, length),
                    output_hidden_states=True,
                )
                hidden = out.hidden_states
                for ci, hid in enumerate(self.cont_layers):
                    z = F.normalize(hidden[hid][:, 0], dim=-1)
                    for cls_id in labels.unique().tolist():
                        lyr = self.class_to_layer.get(int(cls_id), NON_LAYER)
                        if lyr == NON_LAYER:
                            continue
                        mask = labels == cls_id
                        sums[ci, lyr] += z[mask].sum(0)
                        counts[ci, lyr] += mask.sum()
        valid = counts > 0
        centroids = torch.zeros_like(sums)
        centroids[valid] = (sums[valid] / counts[valid].unsqueeze(-1)).to(dtype)
        self.cont_layer_centroids = F.normalize(centroids, dim=-1).detach()
        self.model.train()

    # -- layer-aware loss ---------------------------------------------------

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
            self.hpa_lambda != 0
            and self.cont_layers
            and self.cont_layer_centroids is not None
            and labels is not None
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
            hidden = bert_out.hidden_states
            layers = torch.tensor(
                [self.class_to_layer.get(int(c), NON_LAYER) for c in labels.tolist()],
                device=device,
            )
            valid = layers != NON_LAYER
            layer_terms: list[torch.Tensor] = []
            if bool(valid.any()):
                lv = layers[valid]
                for ci, lyr_idx in enumerate(self.cont_layers):
                    z = F.normalize(hidden[lyr_idx][:, 0], dim=-1)[valid]
                    sims = []
                    for zi, lyr in zip(z, lv.tolist()):
                        cands = [i for i in (lyr - 1, lyr + 1) if 0 <= i < self.n_layers]
                        if cands:
                            c = self.cont_layer_centroids[ci][cands]
                            sims.append((zi @ c.t()).mean())
                    if sims:
                        layer_terms.append(-torch.stack(sims).mean())
            if layer_terms:
                l_cont = torch.stack(layer_terms).mean()
                loss = loss_cls + self.hpa_lambda * l_cont
                if self.state.global_step % 50 == 0:
                    self.log(
                        {
                            "L_cls": float(loss_cls.detach()),
                            "L_cont": float(l_cont.detach()),
                            "hpa_loss": float(loss.detach()),
                            "n_cont_layers": len(layer_terms),
                        }
                    )
            else:
                loss = loss_cls
        return (loss, bert_out) if return_outputs else loss

    # -- damp: per-layer LR groups (headfreeze with --upper-lr-scale<1) -----

    def _is_upper_param(self, name: str) -> bool:
        """True for encoder.layer[i] params with hidden-state index >= threshold."""
        m = re.match(r"bert\.encoder\.layer\.(\d+)", name)
        if not m:
            return False  # embeddings / pooler / classifier head -> full lr
        return int(m.group(1)) >= (self.freeze_from_hidden - 1)

    def create_optimizer(self):
        if not self.damp_upper or self.freeze_upper:
            return super().create_optimizer()
        if self.optimizer is None:
            opt_model = self.model_wrapped if hasattr(self, "model_wrapped") else self.model
            decay_parameters = self.get_decay_parameter_names(opt_model)
            early_decay, early_nodecay, upper_decay, upper_nodecay = [], [], [], []
            for n, p in opt_model.named_parameters():
                if not p.requires_grad:
                    continue
                if self._is_upper_param(n):
                    (upper_decay if n in decay_parameters else upper_nodecay).append(p)
                else:
                    (early_decay if n in decay_parameters else early_nodecay).append(p)
            lr_full = self.args.learning_rate
            lr_upper = lr_full * self.upper_lr_scale
            groups = [
                {"params": early_decay, "weight_decay": self.args.weight_decay, "lr": lr_full},
                {"params": early_nodecay, "weight_decay": 0.0, "lr": lr_full},
                {"params": upper_decay, "weight_decay": self.args.weight_decay, "lr": lr_upper},
                {"params": upper_nodecay, "weight_decay": 0.0, "lr": lr_upper},
            ]
            optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(
                self.args, opt_model
            )
            optimizer_kwargs.pop("params", None)
            self.optimizer = optimizer_cls(groups, **optimizer_kwargs)
            print(
                f"[HPA] DAMP upper layers lr={lr_upper:.2e} (scale={self.upper_lr_scale}); "
                f"groups: early_decay={len(early_decay)} early_nodecay={len(early_nodecay)} "
                f"upper_decay={len(upper_decay)} upper_nodecay={len(upper_nodecay)}",
                flush=True,
            )
        return self.optimizer


class HPAEpochBeginCallback(TrainerCallback):
    """Recompute the per-cont-layer centroids at the start of every epoch.

    transformers 4.57 dispatches ``on_epoch_begin`` on registered
    TrainerCallback OBJECTS, never on the Trainer itself — the inherited
    ``CAFTTrainer.on_epoch_begin`` is dead code, so without this callback the
    LWC continuity term silently never runs (the P0 bug fixed 2026-08-05,
    mirroring the ordinal.py fix). Registered in :func:`build_hpa_trainer`;
    invokes the (ragged-safe) ``HPATrainer._precompute_centroids`` before each
    epoch exactly as the CAFT design intended ("pre-computed over the full
    training set at the start of each epoch").
    """

    def __init__(self, trainer: HPATrainer):
        self.trainer = trainer

    def on_epoch_begin(self, args, state, control, **kwargs):
        # lambda==0 (plain CE under the chosen layer policy) never consumes the
        # centroids; skipping the full-epithelial forward pass keeps the naive
        # arm's runtime identical to the CAFT naive arm.
        if self.trainer.hpa_lambda != 0 and self.trainer.cont_layers:
            self.trainer._precompute_centroids()


def build_hpa_trainer(
    model,
    train_data,
    id_class_dict: dict,
    mode: str,
    hpa_lambda: float,
    training_args_kwargs: dict,
    output_directory: str | Path,
    cont_layers: str | list[int] | None = None,
    final_cont: bool = False,
    freeze_upper: bool = False,
    freeze_from_hidden: int = 5,
    upper_lr_scale: float = 1.0,
    token_dictionary_file: str | Path | None = None,
    eval_data=None,
) -> HPATrainer:
    """Build the HPATrainer; mirrors build_caft_trainer (same collator).

    Registers :class:`HPAEpochBeginCallback` so the per-cont-layer continuity
    centroids are recomputed each epoch (without it the LWC term silently
    never runs — P0 fix 2026-08-05; see the module note).
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
    trainer = HPATrainer(
        model=model,
        args=args,
        data_collator=collator,
        train_dataset=train_data,
        eval_dataset=eval_data,
        mode=mode,
        hpa_lambda=hpa_lambda,
        cont_layers=cont_layers,
        final_cont=final_cont,
        freeze_upper=freeze_upper,
        freeze_from_hidden=freeze_from_hidden,
        upper_lr_scale=upper_lr_scale,
        class_to_layer=class_to_layer_map(id_class_dict),
        token_dictionary_file=token_dictionary_file,
        compute_metrics=None,
    )
    trainer.add_callback(HPAEpochBeginCallback(trainer))
    return trainer
