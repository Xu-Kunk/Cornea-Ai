#!/usr/bin/env python
"""EnergyAnchor-GF: Energy-model A1 arm — MMD anchor to frozen pretrained
differentiation-stage centroids (this study energy-model extension).

Loss:
    L = L_cls + lambda_e * mean_s  MMD^2(z_ft_penult^{(s)}, c_ref^{(s)})
    with RBF kernel exp(-||x-y||^2 / (2 sigma^2)); sigma = median-heuristic
    bandwidth from the frozen reference subsample unless --kernel-bandwidth.
    Reference centroids: frozen (freshly-loaded pretrained) Geneformer
    penultimate hidden-state (default 11) [CLS] embeddings, per stage
    (LSC/TAC/basal/differentiated), computed ONCE over the full train set.
    Non-epithelial cells keep L_cls only (masked out of the anchor), exactly
    like CAFT masks its continuity term.

Pipeline mirrors scripts/caft_gf.py / scripts/run_hp_aft.py EXACTLY (same
tokenizer, same donor split, same Classifier.prepare_data, same HF training
arguments: epochs=3, per-device batch 8 x grad-accum 4 = 32 eff, lr 5e-5,
bf16, warmup 500, wd 0.001, group_by_length); only the loss placement differs
(corneal_fm.energy_anchor.EnergyAnchorTrainer). Tokenization + donor-disjoint
preparation helpers are imported from caft_gf.py (read-only reuse).

Each run writes RUN_STATUS.txt (stage markers) and ENERGY_ANCHOR_DONE (on
success) into its save dir (save/energy_anchor_gf_lam<lambda_e>_seed<seed>_<ts>/);
on crash the full traceback is written to ERROR_TRACEBACK.txt.

Usage:
    python scripts/run_energy_anchor.py --energy-anchor --lambda-energy 0.5 \
        --data_file corneal_100k_donor_split.h5ad --epochs 3 --seed 42 \
        --strict-fair --split_manifest artifacts/paper1_fair_split/manifest.json
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

# GB10 unified-memory family fix (same as the axis arms): expandable segments
# let the caching allocator return long-lived precompute/training segments to
# the driver, avoiding the NVRM IOMMU/vmalloc exhaustion (memdesc.c:1359) that
# SIGKILLed this arm's reference precompute. Must be set before the first CUDA
# allocation.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from caft_gf import make_layer_tokenized, validate_strict_fair_args  # read-only reuse
from datasets import load_from_disk

from corneal_fm.caft import load_pretrained_classifier
from corneal_fm.config import archive_script, get_save_dir, load_config, set_seed
from corneal_fm.energy_anchor import (
    DEFAULT_ANCHOR_HIDDEN,
    build_energy_anchor_trainer,
)
from corneal_fm.utils import check_gpu_memory, get_logger

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("CORNEAL_FM_DATA", str(PROJECT_ROOT / "data")))
DEFAULT_CONFIG = "config/default_train_args.yml"
# A1 arms train on the donor-disjoint train split only (same as the D5 ordinal
# / HPA-FT arms) so results are directly comparable with the locked CAFT/naive
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
@click.option("--epochs", default=None, type=int, help="Training epochs (this study fair protocol = 3).")
@click.option("--batch_size", default=None, type=int, help="Per-device batch size.")
@click.option("--lr", default=None, type=float, help="Learning rate.")
@click.option("--energy-anchor/--no-energy-anchor", "energy_anchor", default=True,
              help="Enable the A1 MMD anchor term (default on; --no-energy-anchor "
                   "with --lambda-energy 0 = plain softmax CE control).")
@click.option("--lambda-energy", "lambda_energy", default=0.5, type=float,
              help="Anchor loss weight λ_e (0 = plain CE under the same pipeline).")
@click.option("--kernel-bandwidth", "kernel_bandwidth", default=None, type=float,
              help="RBF bandwidth σ in exp(-||x-y||^2/(2σ^2)). Default: median "
                   "heuristic over the frozen reference embedding subsample.")
@click.option("--anchor-hidden", "anchor_hidden", default=DEFAULT_ANCHOR_HIDDEN, type=int,
              help="Hidden-state index whose [CLS] embedding is anchored "
                   "(default 11 = penultimate, the head-proximal layer).")
@click.option("--seed", default=42, type=int, help="Random seed.")
@click.option("--strict-fair", "--strict_fair", "strict_fair", is_flag=True, default=False,
              help="Enable this study donor-disjoint fail-closed mode.")
@click.option("--split_manifest", default=None, type=click.Path(dir_okay=False),
              help="this study manifest generated by prepare_paper1_fair_split.py.")
@click.option("--prepared_train", default=None, type=click.Path(exists=False),
              help="Optional already-prepared Geneformer train dataset (strict mode).")
def main(
    config: str | None,
    data_file: str,
    epochs: int | None,
    batch_size: int | None,
    lr: float | None,
    energy_anchor: bool,
    lambda_energy: float,
    kernel_bandwidth: float | None,
    anchor_hidden: int,
    seed: int,
    strict_fair: bool,
    split_manifest: str | None,
    prepared_train: str | None,
) -> None:
    """EnergyAnchor-GF fine-tuning: tokenize-with-layer -> prepare_data -> EnergyAnchorTrainer."""
    manifest = None
    manifest_hash = None
    if strict_fair:
        manifest, manifest_hash = validate_strict_fair_args(
            split_manifest, data_file, epochs, prepared_train
        )
    if not energy_anchor:
        lambda_energy = 0.0  # --no-energy-anchor forces the plain-CE control

    set_seed(seed)
    cfg = load_config(config or DEFAULT_CONFIG)
    mp = cfg["model_parameters"]
    epochs = epochs if epochs is not None else int(mp.get("epochs", 10))
    batch_size = batch_size if batch_size is not None else int(mp.get("batch_size", 16))
    lr = lr if lr is not None else float(mp.get("learning_rate", 5e-5))

    save_dir = get_save_dir("energy_anchor_gf", f"lam{lambda_energy}_seed{seed}")
    save_dir.mkdir(parents=True, exist_ok=True)
    _marker(
        save_dir,
        f"START energy_anchor={energy_anchor} lambda_e={lambda_energy} "
        f"kernel_bandwidth={kernel_bandwidth or 'median-heuristic'} "
        f"anchor_hidden={anchor_hidden} epochs={epochs} bs={batch_size} lr={lr} "
        f"seed={seed} strict_fair={strict_fair}",
    )
    archive_script(__file__, save_dir)

    try:
        _run_energy_anchor(
            save_dir, data_file, epochs, batch_size, lr,
            lambda_energy, kernel_bandwidth, anchor_hidden, seed,
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


def _run_energy_anchor(
    save_dir, data_file, epochs, batch_size, lr,
    lambda_energy, kernel_bandwidth, anchor_hidden, seed,
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

        # (2) prepare_data (31 author_cell_type classes; stage axis derived
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

    # (3) EnergyAnchor training — same bf16-weights setup as caft_gf.py; only
    # the loss placement differs. The trainer computes the frozen reference
    # centroids at init from this freshly-loaded pretrained model.
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = load_pretrained_classifier(
        _locate_pretrained(), num_classes=len(idc), torch_dtype=torch_dtype
    )
    # Move the frozen model to GPU NOW (family fix, mirrors method_arm_gf.py):
    # EnergyAnchorTrainer.__init__ computes the frozen reference centroids by
    # forwarding the model over ALL epithelial training cells, and
    # load_pretrained_classifier() leaves the model on CPU (from_pretrained
    # default). Without this the ~35k-cell reference forward ran on CPU for
    # hours with the GPU idle, and under GB10 unified-memory pressure the
    # driver's IOMMU/vmalloc failure SIGKILLed the run (the per-run log dies
    # right after the model-load warning, rc!=0, no traceback).
    if torch.cuda.is_available():
        model = model.to("cuda")
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
    trainer = build_energy_anchor_trainer(
        model, train_data, idc,
        anchor_lambda=lambda_energy,
        anchor_hidden=anchor_hidden,
        kernel_bandwidth=kernel_bandwidth,
        training_args_kwargs=targs, output_directory=str(Path(save_dir) / "model"),
    )
    # Release the reference precompute's reserved GPU pool before training
    # (family fix, mirrors method_arm_gf.py: on GB10 the caching allocator's
    # retained segments push the kernel toward its OOM watermark).
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    _marker(
        save_dir,
        f"REFERENCE anchor_hidden={trainer.anchor_hidden} "
        f"sigma={trainer.kernel_bandwidth:.4f} "
        f"stage_counts={_stage_counts(trainer)}",
    )
    t0 = time.time()
    trainer.train()
    elapsed = (time.time() - t0) / 60
    _marker(save_dir, f"TRAINED elapsed_min={elapsed:.2f}")
    logger.info("EnergyAnchor training done in %.1f min", elapsed)

    # (4) provenance
    with open(Path(save_dir) / "energy_anchor_args.json", "w") as fh:
        json.dump(
            {"lambda_energy": lambda_energy,
             "anchor_hidden": trainer.anchor_hidden,
             "kernel_bandwidth": trainer.kernel_bandwidth,
             "kernel_bandwidth_requested": kernel_bandwidth,
            "stage_counts_train": _stage_counts(trainer),
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
    with open(Path(save_dir) / "ENERGY_ANCHOR_DONE", "w") as fh:
        fh.write(datetime.now().isoformat(timespec="seconds"))
    _marker(save_dir, "DONE")
    logger.info("✅ EnergyAnchor (λ_e=%.3f) complete -> %s", lambda_energy, save_dir)


def _stage_counts(trainer) -> list[int]:
    """Per-stage epithelial cell counts from the train set (provenance only)."""
    ds = trainer.train_dataset
    if ds is None:
        return []
    labels_all = list(ds["label"])
    counts = [0, 0, 0, 0]
    for c in labels_all:
        st = trainer.class_to_stage.get(int(c), -1)
        if st != -1:
            counts[st] += 1
    return counts


if __name__ == "__main__":
    main()
