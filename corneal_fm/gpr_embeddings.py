"""Checkpoint resolution and embedding extraction for GPR evaluation."""

from __future__ import annotations

import hashlib
import importlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray

from corneal_fm.gpr_metrics import IncompatibleMetricSpaceError


def resolve_checkpoint(checkpoint: str) -> str:
    path = Path(checkpoint).expanduser()
    if path.is_dir():
        candidates = sorted(path.glob("checkpoint-*"))
        if candidates:
            return str(candidates[-1].resolve())
    return str(path.resolve())


def load_model(checkpoint: str, num_classes: int = 20):
    transformers = importlib.import_module("transformers")
    model_class = getattr(transformers, "BertForSequenceClassification")
    model = model_class.from_pretrained(
        resolve_checkpoint(checkpoint),
        num_labels=num_classes,
    )
    if torch.cuda.is_available():
        model = model.cuda()
    return model


def load_from_disk(path: str):
    datasets_module = importlib.import_module("datasets")
    loader = getattr(datasets_module, "load_from_disk")
    return loader(path)


def extract_embeddings(
    model,
    dataset,
    pad_id: int = 0,
    batch: int = 32,
    use_cls: bool = True,
) -> NDArray[np.float32]:
    model.eval()
    device = next(model.parameters()).device
    embeddings = []
    with torch.no_grad():
        for start in range(0, len(dataset), batch):
            batch_data = dataset[start : start + batch]
            sequences = [list(ids) for ids in batch_data["input_ids"]]
            length = batch_data.get("length")
            if length is None:
                length = [len(sequence) for sequence in sequences]
            lengths = torch.as_tensor(length, device=device)
            max_length = int(lengths.max().item()) if sequences else 0
            padded = torch.full(
                (len(sequences), max_length),
                pad_id,
                dtype=torch.long,
                device=device,
            )
            for row, token_ids in enumerate(sequences):
                padded[row, : len(token_ids)] = torch.as_tensor(
                    token_ids,
                    dtype=torch.long,
                    device=device,
                )
            attention = torch.arange(max_length, device=device)[None, :] < lengths[:, None]
            encoder = getattr(model, "bert", model)
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                hidden = encoder(input_ids=padded, attention_mask=attention).last_hidden_state
            pooled = (
                hidden[:, 0]
                if use_cls
                else (hidden.float() * attention.float()[..., None]).sum(1)
                / lengths.float().clamp(min=1)[:, None]
            )
            embeddings.append(F.normalize(pooled.float(), dim=-1).cpu().numpy())
    return np.concatenate(embeddings, axis=0)


def dataset_identity(dataset, tokenized_path: str) -> tuple[str, str]:
    resolved_path = str(Path(tokenized_path).expanduser().resolve())
    fingerprint = getattr(dataset, "_fingerprint", "")
    if fingerprint:
        return resolved_path, f"huggingface:{fingerprint}"
    layers = np.asarray(dataset["layer"], dtype=np.int64)
    digest = hashlib.sha256()
    digest.update(layers.tobytes())
    digest.update(str(len(dataset)).encode())
    return resolved_path, f"sha256:{digest.hexdigest()}"


def require_layers(dataset, tokenized_path: str) -> NDArray[np.int64]:
    if "layer" not in dataset.column_names:
        raise IncompatibleMetricSpaceError(
            f"layer column missing in {tokenized_path}; use the established author_cell_type mapping",
        )
    return np.asarray(dataset["layer"], dtype=np.int64)
