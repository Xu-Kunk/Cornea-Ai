"""CAFT — Continuity-Aware Fine-Tuning (Methods §4.3).

L = L_cls + λ · L_continuity
L_continuity(x) = -mean(cosine_sim(z_CLS(x), centroid_{layer_adj(x)}))

Layer centroids are pre-computed (detached) over the full training set at the
start of each epoch; L_continuity only flows gradient through the cell
embedding z, never through the centroids (stop-gradient formulation, Methods
§4.3). Cells mapped to layer -1 (non-epithelial) keep L_cls but contribute no
continuity term. Corneal epithelial axis: LSC → TAC → basal → wing → squamous.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import BertForSequenceClassification, Trainer, TrainingArguments

LAYER_ORDER = ["LSC", "TAC", "basal", "wing", "squamous"]
NON_LAYER = -1

# author_cell_type -> differentiation layer (Methods §4.4 Method B). PI-ratified
# 2026-07-23 (artifacts/pipeline-20260723/S3/mapping_ratification.md): Epi-PMC/Epi-TDC
# corrected (the earlier code had them reversed vs the KRT3/KRT12 gradient).
CELL_TYPE_TO_LAYER = {
    "LPC": 0,
    "Fetal_Epi/LSC/LPC": 0,
    "Limbus_Epi-C1": 1,
    "Limbus_Epi-C2": 1,
    "Epi-PMC": 2,
    "Epi-TDC/PMC": 3,
    "Epi-TDC": 4,
    "Unknown Epithelium": 4,
    "Conj-Epi": -1,
    "Unknown-DCN+": -1,
}
DEFAULT_PAD_ID = 0


def author_cell_type_to_layer(name: str) -> int:
    return CELL_TYPE_TO_LAYER.get(name, NON_LAYER)


def class_to_layer_map(id_class_dict: dict) -> dict:
    """Map numeric class id -> layer index using author_cell_type names."""
    m = {}
    for cls_id, cls_name in id_class_dict.items():
        m[int(cls_id)] = author_cell_type_to_layer(str(cls_name))
    return m


class CAFTTrainer(Trainer):
    def __init__(
        self,
        caft_lambda: float,
        class_to_layer: dict,
        n_layers: int = 5,
        token_dictionary_file: str | Path | None = None,
        centroid_batch: int = 512,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.caft_lambda = caft_lambda
        self.class_to_layer = class_to_layer
        self.n_layers = n_layers
        self.centroid_batch = centroid_batch
        self.layer_centroids: torch.Tensor | None = None
        self.pad_id = self._resolve_pad_id(token_dictionary_file)

    def _resolve_pad_id(self, token_dictionary_file):
        if token_dictionary_file is None:
            try:
                from geneformer.classifier import TOKEN_DICTIONARY_FILE

                token_dictionary_file = TOKEN_DICTIONARY_FILE
            except Exception:
                return DEFAULT_PAD_ID
        try:
            with open(token_dictionary_file, "rb") as f:
                td = pickle.load(f)
            return int(td["<pad>"]) if "<pad>" in td else DEFAULT_PAD_ID
        except Exception:
            return DEFAULT_PAD_ID

    # -- [CLS]-equivalent embedding (Methods §4.3) -------------------------
    # Geneformer tokenizes as [<cls>=2, genes..., <eos>=3, <pad>=0 ...] with
    # right-padding, so the [CLS] representation is at position 0.
    def _embed(self, input_ids: torch.Tensor, length: torch.Tensor) -> torch.Tensor:
        attn = self._attn_mask(input_ids, length)
        outputs = self.model.bert(input_ids=input_ids, attention_mask=attn)
        return outputs.last_hidden_state[:, 0]

    def _attn_mask(self, input_ids: torch.Tensor, length: torch.Tensor) -> torch.Tensor:
        device = input_ids.device
        return torch.arange(input_ids.shape[1], device=device)[None, :] < length[:, None]

    def on_epoch_begin(self, args, state, control, **kw):
        self._precompute_centroids()

    def _precompute_centroids(self):
        ds = self.train_dataset
        n = len(ds)
        if n == 0:
            return
        device = self.model.device
        self.model.eval()
        sums = torch.zeros(self.n_layers, self.model.config.hidden_size, device=device)
        counts = torch.zeros(self.n_layers, device=device)
        with torch.no_grad():
            for s in range(0, n, self.centroid_batch):
                batch = ds[s : s + self.centroid_batch]
                ids = torch.tensor(batch["input_ids"], device=device)
                length = torch.as_tensor(
                    batch.get("length", ids.ne(self.pad_id).sum(1).tolist()), device=device
                )
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
        centroids[valid] = sums[valid] / counts[valid].unsqueeze(1)
        self.layer_centroids = F.normalize(centroids, dim=-1).detach()
        self.model.train()

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if getattr(self, "_caft_dbg", 0) < 3:
            print(f"[CAFT] inputs keys={list(inputs.keys())}", flush=True)
            self._caft_dbg = getattr(self, "_caft_dbg", 0) + 1
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
        logits = model.classifier(model.dropout(bert_out.pooler_output))
        if labels is not None:
            loss_cls = F.cross_entropy(logits, labels)
        else:
            loss_cls = F.cross_entropy(
                logits, torch.zeros(input_ids.size(0), dtype=torch.long, device=device)
            )
        if self.caft_lambda == 0 or self.layer_centroids is None or labels is None:
            loss = loss_cls
        else:
            z_cls = F.normalize(bert_out.last_hidden_state[:, 0], dim=-1)
            layers = torch.tensor(
                [self.class_to_layer.get(int(c), NON_LAYER) for c in labels.tolist()],
                device=device,
            )
            valid = layers != NON_LAYER  # noqa: F821
            if not bool(valid.any()):
                loss = loss_cls
            else:
                zv = z_cls[valid]
                lv = layers[valid]
                adj_pairs = []
                for lyr in lv.tolist():
                    cands = [i for i in (lyr - 1, lyr + 1) if 0 <= i < self.n_layers]
                    if cands:
                        adj_pairs.append(cands)
                if adj_pairs:
                    sims = []
                    for row, (zi, cands) in enumerate(zip(zv, adj_pairs)):
                        c = self.layer_centroids[cands]
                        sims.append((zi @ c.t()).mean())
                    L_cont = -torch.stack(sims).mean()
                    loss = loss_cls + self.caft_lambda * L_cont
                    if self.state.global_step % 50 == 0:
                        self.log(
                            {
                                "L_cls": float(loss_cls),
                                "L_cont": float(L_cont),
                                "caft_loss": float(loss),
                            }
                        )
                else:
                    loss = loss_cls
        return (loss, bert_out) if return_outputs else loss


def load_pretrained_classifier(
    model_directory: str | Path,
    num_classes: int,
    torch_dtype: torch.dtype | None = None,
) -> BertForSequenceClassification:
    """Load Geneformer as a sequence-classification model.

    ``torch_dtype`` is forwarded to ``from_pretrained``. Pass ``torch.bfloat16``
    to keep the weights in bf16: on the GB10 unified-memory architecture the
    batch-8 transformer forward is weight-bandwidth-bound, so bf16 weights train
    ~2.5x faster than the fp32 master weights that ``bf16=True`` autocast would
    otherwise use (measured 2.2 vs 5.6 s/micro-batch). Default ``None`` keeps the
    historical fp32 behaviour.
    """
    kwargs: dict[str, Any] = {}
    if torch_dtype is not None:
        kwargs["dtype"] = torch_dtype
    model = BertForSequenceClassification.from_pretrained(
        str(model_directory),
        num_labels=num_classes,
        output_attentions=False,
        id2label={i: str(i) for i in range(num_classes)},
        label2id={str(i): i for i in range(num_classes)},
        **kwargs,
    )
    return model


def build_caft_trainer(
    model,
    train_data,
    id_class_dict: dict,
    caft_lambda: float,
    training_args_kwargs: dict,
    output_directory: str | Path,
    token_dictionary_file: str | Path | None = None,
    eval_data=None,
) -> CAFTTrainer:
    from geneformer.collator_for_classification import DataCollatorForCellClassification

    # geneformer (jkobject fork) names the cell collator DataCollatorForCellClassification
    # and takes the loaded token dictionary, not a file path (mirrors Classifier in
    # classifier.py which does DataCollatorForCellClassification(token_dictionary=...)).
    if token_dictionary_file is None:
        from geneformer.classifier import TOKEN_DICTIONARY_FILE as token_dictionary_file
    with open(token_dictionary_file, "rb") as _fh:
        token_dictionary = pickle.load(_fh)
    collator = DataCollatorForCellClassification(token_dictionary=token_dictionary)
    args = TrainingArguments(output_dir=str(output_directory), **training_args_kwargs)
    return CAFTTrainer(
        model=model,
        args=args,
        data_collator=collator,
        train_dataset=train_data,
        eval_dataset=eval_data,
        caft_lambda=caft_lambda,
        class_to_layer=class_to_layer_map(id_class_dict),
        token_dictionary_file=token_dictionary_file,
        compute_metrics=None,
    )
