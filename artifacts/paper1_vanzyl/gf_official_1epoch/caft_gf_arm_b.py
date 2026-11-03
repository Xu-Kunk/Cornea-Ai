#!/usr/bin/env python
"""CAFT-GF: Continuity-Aware Fine-Tuning of Geneformer V2-104M (Methods §4.3).

Arm B variant (PREREG-2026-08-11 §9, 2026-08-18): GF official-budget control.
Changes vs the LOCKED caft_gf.py (sha256 bdeacd18b4d61ac190ab2b5eed3b6748a
41ac935fe7d8aee1cbe2511bb9e763d) are EXACTLY TWO, both input-contract only:
  A1. validate_strict_fair_args accepts --epochs 1 in addition to 3 (the
      strict-fair gate hard-coded 3; Arm B = GF official 1-epoch budget).
  A2. the --data_file absolute-path equality against the manifest's
      input_paths.train_h5ad (a machine-specific absolute path) is relaxed to
      a basename comparison; data identity is verified separately via the
      manifest checksums.train_h5ad sha256 (bc357c7d...) in the ops card.
No training-science code (loss, model, data, split, optimizer, scheduler,
checkpointing) was modified. Verify: `diff caft_gf.py caft_gf_arm_b.py`.

Trains a cell-type classifier:  L = L_cls + λ · L_continuity
L_continuity(x) = -mean(cos(z_CLS, centroid_{layer_adj(x)}))  (stop-gradient centroids).

Pipeline: (1) re-tokenize cells with an added 'layer' column (derived from
author_cell_type via the §4.4 Method B mapping); (2) Classifier.prepare_data
(label + 8/2 split); (3) CAFTTrainer. The standard baseline is λ=0.

Each run writes RUN_STATUS.txt (stage markers) and CAFT_DONE (on success) into
its save dir; on crash the full traceback is written to ERROR_TRACEBACK.txt, so
outcome is verifiable without relying on stdout capture.
"""

from __future__ import annotations

import glob
import json
import os
import pickle
import shutil
import sys
import tempfile
import time
import traceback
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import click
import numpy as np
import torch
import anndata as ad
from datasets import load_from_disk

from corneal_fm.caft import (
    LAYER_ORDER,
    author_cell_type_to_layer,
    build_caft_trainer,
    load_pretrained_classifier,
)
from corneal_fm.config import archive_script, get_save_dir, load_config, set_seed
from corneal_fm.utils import check_gpu_memory, get_logger

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("CORNEAL_FM_DATA", str(PROJECT_ROOT / "data")))
DEFAULT_CONFIG = "config/default_train_args.yml"
DEFAULT_DATA_FILE = "corneal_100k_subsample.h5ad"
PER_DEVICE_TRAIN_BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 4
_HF_SNAP_GLOB = os.path.expanduser(
    "~/.cache/huggingface/hub/models--ctheodoris--Geneformer/snapshots/*/Geneformer-V2-104M"
)


def _marker(save_dir, text):
    """Append a datestamped status line to RUN_STATUS.txt (verifiable post-hoc)."""
    with open(Path(save_dir) / "RUN_STATUS.txt", "a") as f:
        f.write(f"[{datetime.now().isoformat(timespec='seconds')}] {text}\n")


def _locate_pretrained() -> str:
    hits = glob.glob(_HF_SNAP_GLOB)
    if not hits:
        logger.error("Geneformer-V2-104M weights not found in HF cache.")
        sys.exit(1)
    return hits[0]


def _manifest_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_strict_fair_args(
    split_manifest: str | None,
    data_file: str,
    epochs: int | None,
    prepared_train: str | None = None,
) -> tuple[dict, str]:
    """Fail closed before a strict run can select the legacy cell-level split."""
    if not split_manifest:
        raise ValueError("strict fair mode requires --split_manifest")
    manifest_path = Path(split_manifest).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"strict fair manifest not found: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid strict fair manifest: {manifest_path}") from exc
    if manifest.get("mode") != "strict_fair":
        raise ValueError(f"strict fair manifest mode must be 'strict_fair', got {manifest.get('mode')!r}")
    if manifest.get("protocol") != "paper1_fair":
        raise ValueError(f"strict fair manifest protocol must be 'paper1_fair', got {manifest.get('protocol')!r}")
    overlap = manifest.get("donor_overlap", manifest.get("donor_overlap_count"))
    if isinstance(overlap, dict):
        overlap = overlap.get("count", overlap.get("overlap_count"))
    if overlap != 0:
        raise ValueError(f"strict fair manifest donor overlap must be 0, got {overlap!r}")
    if Path(data_file).name == DEFAULT_DATA_FILE:
        raise ValueError(
            "strict fair mode refuses corneal_100k_subsample.h5ad; use the donor-disjoint train H5AD"
        )
    if epochs is None:
        raise ValueError("strict fair mode requires explicit --epochs (Paper 1 baseline is 3)")
    if epochs not in (1, 3):
        # Arm B (PREREG §9): GF official budget = 1 epoch; 3 = Paper 1 baseline.
        raise ValueError(f"strict fair Paper 1 baseline requires --epochs 3 (Arm B: 1), got {epochs}")
    if prepared_train is not None and not Path(prepared_train).exists():
        raise FileNotFoundError(f"prepared training dataset not found: {prepared_train}")
    expected_train = manifest.get("input_paths", {}).get("train_h5ad")
    if expected_train and Path(data_file).name != Path(expected_train).name:
        # A2 (Arm B): the manifest stores a machine-specific absolute path;
        # compare basenames; content identity verified via manifest
        # checksums.train_h5ad sha256 in the ops card.
        raise ValueError("--data_file does not match the donor-disjoint train path in --split_manifest")
    return manifest, _manifest_sha256(manifest_path)


def make_layer_tokenized(src_h5ad: Path, out_h5ad: Path) -> dict:
    """Promote raw counts, add 'layer' (from author_cell_type), tokenize with layer preserved."""
    from geneformer import TranscriptomeTokenizer
    from corneal_fm import gf_compat
    gf_compat.install()

    logger.info("Loading %s ...", src_h5ad.name)
    a = ad.read_h5ad(str(src_h5ad))
    sub = a  # full dataset by default
    if sub.raw is not None:
        sub.X = sub.raw.X.copy()
        sub.raw = None
    sub.obs = sub.obs.reset_index(drop=True)
    # Keep the Ensembl-ID var.index intact: TranscriptomeTokenizer uses
    # use_h5ad_index=True (gene ids read from var.index); resetting it to a
    # RangeIndex tokenizes zero cells. Mirrors the proven protocol_04 path.
    var = sub.var.copy()
    if "ensembl_id_collapsed" not in var.columns and "ensembl_id" in var.columns:
        var["ensembl_id_collapsed"] = var["ensembl_id"].values
    sub.var = var
    sub.obs["n_counts"] = np.asarray(sub.X.sum(axis=1)).ravel()
    sub.obs["layer"] = sub.obs["author_cell_type"].map(author_cell_type_to_layer).astype("int64")
    sub.write_h5ad(str(out_h5ad), compression="gzip")
    ld = sub.obs["layer"].tolist()
    layer_dist = {LAYER_ORDER[i]: ld.count(i) for i in range(5)} | {"nonEpi": ld.count(-1)}
    logger.info("layer dist: %s | cells=%d genes=%d", layer_dist, sub.n_obs, sub.n_vars)

    tk = TranscriptomeTokenizer(
        custom_attr_name_dict={
            "cell_type": "cell_type",
            "author_cell_type": "author_cell_type",
            "disease": "disease",
            "tissue": "tissue",
            "donor_id": "donor_id",
            "layer": "layer",
        },
        nproc=4,
        chunk_size=512,
        model_version="V2",
        use_h5ad_index=True,
        collapse_gene_ids=True,
    )
    t0 = time.time()
    tmp_dir = Path(tempfile.mkdtemp(prefix="caft_tk_"))
    tmp_h5ad = tmp_dir / out_h5ad.name
    shutil.copy2(str(out_h5ad), str(tmp_h5ad))
    tk_dir = out_h5ad.parent / (out_h5ad.stem + "_tk")
    tk_dir.mkdir(parents=True, exist_ok=True)
    tk.tokenize_data(
        data_directory=str(tmp_dir),
        output_directory=str(tk_dir),
        output_prefix="cell",
        file_format="h5ad",
    )
    shutil.rmtree(tmp_dir, ignore_errors=True)
    ds_path = str(tk_dir / "cell.dataset")
    n_cells = len(load_from_disk(ds_path))
    logger.info("tokenized %d cells in %.1f s -> %s", n_cells, time.time() - t0, ds_path)
    return {"layer_dist": layer_dist, "cells": n_cells, "dataset": ds_path}


@click.command()
@click.option("--config", default=None, help="YAML config (default: config/default_train_args.yml).")
@click.option("--data_file", default=DEFAULT_DATA_FILE, help="Source h5ad under CORNEAL_FM_DATA.")
@click.option("--epochs", default=None, type=int, help="Training epochs.")
@click.option("--batch_size", default=None, type=int, help="Per-device batch size.")
@click.option("--lr", default=None, type=float, help="Learning rate.")
@click.option("--caft_lambda", default=0.5, type=float, help="Continuity loss weight λ (0 = standard baseline).")
@click.option("--freeze_layers", default=0, type=int, help="Encoder layers to freeze.")
@click.option("--seed", default=42, type=int, help="Random seed.")
@click.option("--strict-fair", "--strict_fair", "strict_fair", is_flag=True, default=False,
              help="Enable Paper 1 donor-disjoint fail-closed mode.")
@click.option("--split_manifest", default=None, type=click.Path(dir_okay=False),
              help="Paper 1 manifest generated by prepare_paper1_fair_split.py.")
@click.option("--prepared_train", default=None, type=click.Path(exists=False),
              help="Optional already-prepared Geneformer train dataset (strict mode).")
def main(
    config: str | None,
    data_file: str,
    epochs: int | None,
    batch_size: int | None,
    lr: float | None,
    caft_lambda: float,
    freeze_layers: int,
    seed: int,
    strict_fair: bool,
    split_manifest: str | None,
    prepared_train: str | None,
) -> None:
    """CAFT-GF fine-tuning: tokenize-with-layer -> prepare_data -> CAFTTrainer."""
    manifest = None
    manifest_hash = None
    if strict_fair:
        manifest, manifest_hash = validate_strict_fair_args(
            split_manifest, data_file, epochs, prepared_train
        )
    set_seed(seed)
    cfg = load_config(config or DEFAULT_CONFIG)
    mp = cfg["model_parameters"]
    epochs = epochs if epochs is not None else int(mp.get("epochs", 10))
    batch_size = batch_size if batch_size is not None else int(mp.get("batch_size", 16))
    lr = lr if lr is not None else float(mp.get("learning_rate", 5e-5))

    save_dir = get_save_dir("caft_gf", f"lam{caft_lambda}_seed{seed}")
    save_dir.mkdir(parents=True, exist_ok=True)
    _marker(save_dir, f"START lam={caft_lambda} epochs={epochs} bs={batch_size} lr={lr} seed={seed} freeze={freeze_layers} strict_fair={strict_fair}")
    archive_script(__file__, save_dir)

    try:
        _run_caft(
            save_dir, data_file, epochs, batch_size, lr, caft_lambda, freeze_layers, seed,
            split_manifest=split_manifest, manifest=manifest, manifest_hash=manifest_hash,
            prepared_train=prepared_train, strict_fair=strict_fair,
        )
    except Exception:
        tb = traceback.format_exc()
        with open(save_dir / "ERROR_TRACEBACK.txt", "w") as fh:
            fh.write(f"[{datetime.now().isoformat(timespec='seconds')}]\n{tb}\n")
        _marker(save_dir, "ERROR\n" + tb)
        logger.error("Run failed; traceback -> %s/ERROR_TRACEBACK.txt", save_dir)
        raise


def _run_caft(
    save_dir, data_file, epochs, batch_size, lr, caft_lambda, freeze_layers, seed,
    *, split_manifest=None, manifest=None, manifest_hash=None, prepared_train=None,
    strict_fair=False,
):
    logger.info("Device GPU mem: %s", check_gpu_memory())
    src = DATA / data_file
    if not src.exists():
        raise FileNotFoundError(f"source h5ad not found: {src}")

    if prepared_train:
        # Strict mode may consume an independently prepared donor-disjoint
        # dataset.  It avoids accidentally rebuilding from a legacy tokenized
        # post-grid artifact; the class dictionary must live beside it.
        prep_train = str(Path(prepared_train).resolve())
        prep_root = Path(prep_train).parent
        idc_path = prep_root / "caft_id_class_dict.pkl"
        if not idc_path.exists():
            raise FileNotFoundError(f"class dictionary missing beside --prepared_train: {idc_path}")
        info = {"layer_dist": {}, "cells": None, "dataset": prep_train}
        ds_path = prep_train
        idc = pickle.load(idc_path.open("rb"))
        _marker(save_dir, f"PREPARED_INPUT dataset={prep_train}")
    else:
        # (1) tokenize with layer
        lay_out = str(Path(save_dir) / "corneal_layer_subset.h5ad")
        info = make_layer_tokenized(src, Path(lay_out))
        ds_path = info["dataset"]
        _marker(save_dir, f"TOKENIZED cells={info['cells']} path={ds_path}")
        logger.info("tokenized dataset: %s (cells=%d)", ds_path, info["cells"])

        # (2) prepare_data
        from geneformer import Classifier

        prep_dir = str(Path(save_dir) / "prepared")
        os.makedirs(prep_dir, exist_ok=True)
        cc = Classifier(
            classifier="cell",
            cell_state_dict={"state_key": "author_cell_type", "states": "all"},
            model_version="V2",
            num_crossval_splits=1,
            nproc=4,
        )
        cc.prepare_data(input_data_file=ds_path, output_directory=prep_dir, output_prefix="caft", test_size=0.2)
        prep_train = sorted(glob.glob(os.path.join(prep_dir, "caft_labeled_train.dataset")))[0]
        idc = pickle.load(open(os.path.join(prep_dir, "caft_id_class_dict.pkl"), "rb"))
    train_data = load_from_disk(prep_train)
    # The geneformer collator tensors every column; keep only what the model/loss
    # consume and drop the string metadata (cell_type, author_cell_type, ...) plus
    # the per-cell 'layer' (continuity uses class_to_layer, GPR reads the raw
    # tokenized dataset, neither needs those columns in the training batch).
    # Keep 'length' so TrainingArguments(group_by_length=True) can group similar-length
    # cells and cut padding waste (official Geneformer recipe uses group_by_length).
    keep = {"input_ids", "length", "label"}
    drop_cols = [c for c in train_data.column_names if c not in keep]
    if drop_cols:
        train_data = train_data.remove_columns(drop_cols)
    _marker(save_dir, f"PREPARED train={len(train_data)} num_classes={len(idc)}")
    logger.info("prepared: train=%d  num_classes=%d", len(train_data), len(idc))

    # (3) CAFT training
    # bf16 WEIGHTS (not fp32 master + autocast): on the GB10 unified-memory GPU
    # the batch-8 forward is weight-bandwidth-bound, so fp32 master weights train
    # ~2.5x slower (5.6 vs 2.2 s/micro-batch). bf16 weights + bf16 autocast (the
    # bf16=True arg below keeps layer-norm/loss in fp32) removes that overhead
    # while preserving CAFT semantics. See artifacts/probe/caft_prof.py.
    model = load_pretrained_classifier(
        _locate_pretrained(), num_classes=len(idc), torch_dtype=torch.bfloat16
    )
    targs = {
        "num_train_epochs": epochs,
        "per_device_train_batch_size": PER_DEVICE_TRAIN_BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "learning_rate": lr,
        "lr_scheduler_type": "linear",
        "warmup_steps": 500,
        "weight_decay": 0.001,
        "logging_steps": 10,
        "save_strategy": "epoch",
        "save_total_limit": 2,
        "eval_strategy": "no",
        "fp16": False,
        "bf16": torch.cuda.is_available(),
        "max_grad_norm": 1.0,
        "gradient_checkpointing": True,
        "dataloader_pin_memory": torch.cuda.is_available(),
        "dataloader_num_workers": 0,
        "load_best_model_at_end": False,
        "remove_unused_columns": False,
        "group_by_length": True,
        "length_column_name": "length",
        "report_to": [],
    }
    trainer = build_caft_trainer(
        model, train_data, idc,
        caft_lambda=caft_lambda, training_args_kwargs=targs, output_directory=str(Path(save_dir) / "model"),
    )
    t0 = time.time()
    trainer.train()
    elapsed = (time.time() - t0) / 60
    _marker(save_dir, f"TRAINED elapsed_min={elapsed:.2f}")
    logger.info("CAFT training done in %.1f min", elapsed)

    # (4) provenance
    with open(Path(save_dir) / "caft_args.json", "w") as fh:
        json.dump(
            {"caft_lambda": caft_lambda, "layer_dist": info["layer_dist"], "num_classes": len(idc),
             "train_cells": len(train_data), "epochs": epochs, "batch_size": batch_size, "lr": lr,
             # Effective training batch provenance (F5): the config-default
             # batch_size above is kept for continuity, but the true optimizer
             # step is per_device_train_batch_size * gradient_accumulation_steps.
             "per_device_train_batch_size": PER_DEVICE_TRAIN_BATCH_SIZE,
             "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
             "effective_batch_size_per_device": PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS,
             "seed": seed, "strict_fair": strict_fair, "split_manifest": split_manifest,
             "manifest_sha256": manifest_hash, "prepared_train": prepared_train,
             "elapsed_min": round(elapsed, 2),
             "log_history_tail": trainer.state.log_history[-3:]},
            fh, indent=2, default=str,
        )
    with open(Path(save_dir) / "CAFT_DONE", "w") as fh:
        fh.write(datetime.now().isoformat(timespec="seconds"))
    _marker(save_dir, "DONE")
    logger.info("✅ CAFT-GF (λ=%.3f) complete -> %s", caft_lambda, save_dir)


if __name__ == "__main__":
    main()
