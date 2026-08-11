#!/usr/bin/env python
"""HPA-FT-GF: Head-Proximal-Aware Fine-Tuning of Geneformer V2-104M (Paper 1 main method).

Loss (mode-dependent):
    L = L_cls + lambda * mean_{c in cont_layers} L_cont_c        (mode=lwc)
    L = L_cls (+ optional continuity) with L5-L12 frozen/damped   (mode=headfreeze)
    both = headfreeze + lwc continuity
    L_cont_c(z) = -mean(cos(z_CLS^{(c)}, centroid^{(c)}_{layer_adj(x)}))  (stop-grad, as CAFT)

Modes:
    lwc        : layer-wise continuity over --cont-layers (default L3-L10)
    headfreeze : freeze (default) or damp (--upper-lr-scale<1) encoder
                 layers at/above --freeze-upper-from (default 5 = L5-L12);
                 only early layers + head trained
    both       : headfreeze + lwc continuity combined (default)

Both variants keep the CAFT-style FINAL-representation continuity as an option
(--final-cont, hidden state 12).

Pipeline mirrors scripts/caft_gf.py / scripts/ordinal_gf.py EXACTLY (same
tokenizer, same donor split, same Classifier.prepare_data, same HF training
arguments); only the loss placement / layer freeze-damp policy differ
(corneal_fm.hp_aft.HPATrainer instead of CAFTTrainer). Tokenization +
donor-disjoint preparation helpers are imported from caft_gf.py (read-only
reuse; caft_gf.py itself is not modified).

Each run writes RUN_STATUS.txt (stage markers) and HPA_DONE (on success) into
its save dir (save/hp_aft_gf_<mode>_lam<lambda>_seed<seed>_<ts>/); on crash
the full traceback is written to ERROR_TRACEBACK.txt.

Usage:
    python scripts/run_hp_aft.py --mode lwc --lambda 0.5 --cont-layers 3-10 \
        --seed 42 --strict-fair --split_manifest artifacts/paper1_fair_split/manifest.json
"""

from __future__ import annotations

import glob
import json
import os
import pickle
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import click
import torch
from caft_gf import make_layer_tokenized, validate_strict_fair_args  # read-only reuse
from datasets import load_from_disk

from corneal_fm.caft import load_pretrained_classifier
from corneal_fm.config import archive_script, get_save_dir, load_config, set_seed
from corneal_fm.hp_aft import (
    MODE_DEFAULT_CONT,
    VALID_MODES,
    build_hpa_trainer,
    parse_cont_layers,
)
from corneal_fm.utils import check_gpu_memory, get_logger

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("CORNEAL_FM_DATA", str(PROJECT_ROOT / "data")))
DEFAULT_CONFIG = "config/default_train_args.yml"
# HPA-FT arms train on the donor-disjoint train split only (same as the D5
# ordinal arm) so results are directly comparable with the locked CAFT/naive
# arms in the layer-wise rho diagnostic.
DEFAULT_DATA_FILE = "corneal_100k_donor_split.h5ad"
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


@click.command()
@click.option("--config", default=None, help="YAML config (default: config/default_train_args.yml).")
@click.option("--data_file", default=DEFAULT_DATA_FILE, help="Source h5ad under CORNEAL_FM_DATA.")
@click.option("--epochs", default=None, type=int, help="Training epochs (Paper 1 fair protocol = 3).")
@click.option("--batch_size", default=None, type=int, help="Per-device batch size.")
@click.option("--lr", default=None, type=float, help="Learning rate.")
@click.option("--mode", default="both", type=click.Choice(list(VALID_MODES)),
              help="HPA-FT variant: lwc | headfreeze | both (default both).")
@click.option("--lambda", "hpa_lambda", default=0.5, type=float,
              help="Continuity loss weight λ (0 = plain CE under the chosen layer policy).")
@click.option("--cont-layers", "cont_layers", default=None,
              help="Hidden-state indices for LWC continuity, e.g. '3-10' or '3,4,5,10' "
                   "(token 'final' = hidden state 12). Default per mode: "
                   f"{MODE_DEFAULT_CONT}.")
@click.option("--final-cont", "final_cont", is_flag=True, default=False,
              help="Also apply the CAFT-style continuity on the FINAL representation "
                   "(hidden state 12), in addition to --cont-layers.")
@click.option("--freeze-upper", "freeze_upper", is_flag=True, default=False,
              help="Fully freeze encoder layers at/above --freeze-upper-from "
                   "(headfreeze mode default; incompatible with --upper-lr-scale<1).")
@click.option("--freeze-upper-from", "freeze_upper_from", default=5, type=int,
              help="Freeze/damp hidden states >= this index (default 5 = L5-L12, the "
                   "Paper 1 collapsed band).")
@click.option("--upper-lr-scale", "upper_lr_scale", default=1.0, type=float,
              help="LR multiplier for upper layers. <1.0 in headfreeze/both mode switches "
                   "from freeze to per-layer-LR damp (default 1.0 = freeze).")
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
    mode: str,
    hpa_lambda: float,
    cont_layers: str | None,
    final_cont: bool,
    freeze_upper: bool,
    freeze_upper_from: int,
    upper_lr_scale: float,
    seed: int,
    strict_fair: bool,
    split_manifest: str | None,
    prepared_train: str | None,
) -> None:
    """HPA-FT-GF fine-tuning: tokenize-with-layer -> prepare_data -> HPATrainer."""
    manifest = None
    manifest_hash = None
    if strict_fair:
        manifest, manifest_hash = validate_strict_fair_args(
            split_manifest, data_file, epochs, prepared_train
        )
    if cont_layers is not None:
        parse_cont_layers(cont_layers)  # fail fast on bad spec
    if freeze_upper and upper_lr_scale < 1.0:
        raise click.UsageError("--freeze-upper and --upper-lr-scale<1 are mutually exclusive")

    set_seed(seed)
    cfg = load_config(config or DEFAULT_CONFIG)
    mp = cfg["model_parameters"]
    epochs = epochs if epochs is not None else int(mp.get("epochs", 10))
    batch_size = batch_size if batch_size is not None else int(mp.get("batch_size", 16))
    lr = lr if lr is not None else float(mp.get("learning_rate", 5e-5))

    save_dir = get_save_dir("hp_aft_gf", f"{mode}_lam{hpa_lambda}_seed{seed}")
    save_dir.mkdir(parents=True, exist_ok=True)
    _marker(
        save_dir,
        f"START mode={mode} lambda={hpa_lambda} cont_layers={cont_layers or 'auto'} "
        f"final_cont={final_cont} freeze_upper={freeze_upper} "
        f"freeze_from_hidden={freeze_upper_from} upper_lr_scale={upper_lr_scale} "
        f"epochs={epochs} bs={batch_size} lr={lr} seed={seed} strict_fair={strict_fair}",
    )
    archive_script(__file__, save_dir)

    try:
        _run_hpa(
            save_dir, data_file, epochs, batch_size, lr,
            mode, hpa_lambda, cont_layers, final_cont, freeze_upper,
            freeze_upper_from, upper_lr_scale, seed,
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


def _run_hpa(
    save_dir, data_file, epochs, batch_size, lr,
    mode, hpa_lambda, cont_layers, final_cont, freeze_upper,
    freeze_upper_from, upper_lr_scale, seed,
    *, split_manifest=None, manifest=None, manifest_hash=None, prepared_train=None,
    strict_fair=False,
):
    logger.info("Device GPU mem: %s", check_gpu_memory())
    src = DATA / data_file
    if not src.exists():
        raise FileNotFoundError(f"source h5ad not found: {src}")

    if prepared_train:
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
        # (1) tokenize with layer (identical to caft_gf.py)
        lay_out = str(Path(save_dir) / "corneal_layer_subset.h5ad")
        info = make_layer_tokenized(src, Path(lay_out))
        ds_path = info["dataset"]
        _marker(save_dir, f"TOKENIZED cells={info['cells']} path={ds_path}")
        logger.info("tokenized dataset: %s (cells=%d)", ds_path, info["cells"])

        # (2) prepare_data (31 author_cell_type classes; layer axis derived
        # from the same mapping at loss time)
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
    # Same column trimming as caft_gf.py (collator re-adds attention_mask/labels).
    keep = {"input_ids", "length", "label"}
    drop_cols = [c for c in train_data.column_names if c not in keep]
    if drop_cols:
        train_data = train_data.remove_columns(drop_cols)
    _marker(save_dir, f"PREPARED train={len(train_data)} num_classes={len(idc)}")
    logger.info("prepared: train=%d  num_classes=%d", len(train_data), len(idc))

    # (3) HPA-FT training — same bf16-weights setup as caft_gf.py; only the
    # loss placement and the freeze/damp policy differ.
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = load_pretrained_classifier(
        _locate_pretrained(), num_classes=len(idc), torch_dtype=torch_dtype
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
    trainer = build_hpa_trainer(
        model, train_data, idc,
        mode=mode, hpa_lambda=hpa_lambda,
        cont_layers=cont_layers, final_cont=final_cont,
        freeze_upper=freeze_upper, freeze_from_hidden=freeze_upper_from,
        upper_lr_scale=upper_lr_scale,
        training_args_kwargs=targs, output_directory=str(Path(save_dir) / "model"),
    )
    t0 = time.time()
    trainer.train()
    elapsed = (time.time() - t0) / 60
    _marker(save_dir, f"TRAINED elapsed_min={elapsed:.2f}")
    logger.info("HPA-FT training done in %.1f min", elapsed)

    # (4) provenance
    with open(Path(save_dir) / "hp_aft_args.json", "w") as fh:
        json.dump(
            {"mode": mode, "hpa_lambda": hpa_lambda,
             "cont_layers": trainer.cont_layers,
             "cont_layers_requested": cont_layers, "final_cont": final_cont,
             "freeze_upper": trainer.freeze_upper,
             "freeze_from_hidden": trainer.freeze_from_hidden,
             "upper_lr_scale": trainer.upper_lr_scale,
             "damp_upper": trainer.damp_upper,
             "layer_dist": info["layer_dist"], "num_classes": len(idc),
             "train_cells": len(train_data), "epochs": epochs, "batch_size": batch_size,
             "lr": lr,
             "per_device_train_batch_size": PER_DEVICE_TRAIN_BATCH_SIZE,
             "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
             "effective_batch_size_per_device": PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS,
             "seed": seed, "strict_fair": strict_fair, "split_manifest": split_manifest,
             "manifest_sha256": manifest_hash, "prepared_train": prepared_train,
             "elapsed_min": round(elapsed, 2),
             "log_history_tail": trainer.state.log_history[-3:]},
            fh, indent=2, default=str,
        )
    with open(Path(save_dir) / "HPA_DONE", "w") as fh:
        fh.write(datetime.now().isoformat(timespec="seconds"))
    _marker(save_dir, "DONE")
    logger.info("✅ HPA-FT (mode=%s λ=%.3f) complete -> %s", mode, hpa_lambda, save_dir)


if __name__ == "__main__":
    main()
