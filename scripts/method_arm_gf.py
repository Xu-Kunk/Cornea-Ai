#!/usr/bin/env python
"""Method-arm-GF: this study loss-swap fine-tuning of Geneformer V2-104M.

Trains one of the three new this study method arms on the locked protocol
(Geneformer V2-104M, 4-class corneal ordinal labels
LSC(0) < TAC(1) < basal(2) < differentiated(3), 3 epochs, eff. batch 32,
seeds 42/123/456, external eval GSE153515):

    --arm axis_ft        L = L_CE + lambda * ||P_A (h - h0)||^2
                         (corneal_fm.axis_ft.AxisFTTrainer; anchor layer
                         configurable, default final hidden state)
    --arm soft_spearman  L = L_CE + lambda * (1 - rho_soft(proj_A(h), y))
                         (corneal_fm.soft_spearman.SoftSpearmanTrainer)
    --arm flow           L = L_CE + lambda * mean adjacent-pair mid-point
                         flow deviation (corneal_fm.flow_consistency)
    --arm ce             L = L_CE only, plain cross-entropy over the active
                         label space (--label-space), no auxiliary term; the
                         trainer is the same CAFTTrainer at lambda=0 the naive
                         arm uses, so this isolates supervision / label set

Label space (``--label-space``, default ``author31`` = the locked behaviour):

* ``author31`` — 31 author_cell_type classes, unchanged for every existing
  invocation (byte-identical defaults, same code path).
* ``epi4`` — the training partition is filtered to epithelial cells only
  (NON_LAYER dropped) and labels collapse to the 4 evaluation stage classes
  (LSC=0 < TAC=1 < basal=2 < differentiated=3; wing and squamous both map to
  differentiated). Only valid with ``--arm ce`` (the CE-over-4-stages
  baseline); the ordinal arms already consume this mapping at loss time.

All three swap ONLY the loss on the shared locked pipeline: same tokenizer /
donor split / Classifier.prepare_data / HF training arguments as
scripts/caft_gf.py / scripts/ordinal_gf.py (imported read-only from
caft_gf.py). The classification head is the same 4-class CE head as the
paper's naive arm; non-epithelial cells (layer == -1) are masked out of the
loss (mirroring ordinal.py).

The frozen differentiation axis ``w`` is computed ONCE from the pretrained
model by corneal_fm.axis_ref and passed to the trainer; it is never
recomputed inside the training loop. The axis definition is selected with
``--axis-mode``:

* ``centroid_pca`` (DEFAULT — the NEW primary, protocol deviation
  2026-08-10): the per-layer-centroid PCA axis fit on ALL training
  epithelial cells, the eval metric's training-side analog (expected
  external alignment rho ~0.777, cos(a_int, a_ext) ~0.818 vs the old
  marker-score axis's cos 0.470). Cached to
  artifacts/paper1_method_arms/axis_ref.json under a mode-aware cache key
  (schema v2) — the retrain consumes the new axis automatically and never
  loads the old marker-score record.
* ``marker_score`` (secondary/reference): the original marker-score
  soft-rank-optimised axis.

The axis_ft arm additionally captures the frozen per-cell h0 embeddings
once before the first optimizer step (AxisFTInitCallback; ~11 min epi-only
forward, then zero per-batch cost).

Each run writes RUN_STATUS.txt (stage markers) and METHOD_ARM_DONE (on
success) into its save dir (save/method_<arm>_lam<lambda>_seed<seed>_<ts>/);
on crash the full traceback is written to ERROR_TRACEBACK.txt.

Usage:
    python scripts/method_arm_gf.py --arm axis_ft --lambda-arm 0.5 \
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
from caft_gf import make_layer_tokenized, validate_strict_fair_args  # read-only reuse
from datasets import load_from_disk

from corneal_fm.axis_ft import build_axis_ft_trainer
from corneal_fm.axis_ref import (
    AXIS_CACHE_FILE,
    AXIS_MODES,
    DEFAULT_AXIS_MODE,
    axis_cache_key,
    compute_axis,
    default_geneformer_dicts,
)
from corneal_fm.caft import (
    NON_LAYER,
    build_caft_trainer,
    class_to_layer_map,
    load_pretrained_classifier,
)
from corneal_fm.config import archive_script, get_save_dir, load_config, set_seed
from corneal_fm.flow_consistency import build_flow_trainer
from corneal_fm.ordinal import N_ORDINAL, class_to_ordinal_map
from corneal_fm.soft_spearman import build_ss_trainer
from corneal_fm.utils import check_gpu_memory, get_logger

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("CORNEAL_FM_DATA", str(PROJECT_ROOT / "data")))
DEFAULT_CONFIG = "config/default_train_args.yml"
# this study method arms train on the donor-disjoint train split only.
DEFAULT_DATA_FILE = "corneal_100k_donor_split.h5ad"
PER_DEVICE_TRAIN_BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 4
_HF_SNAP_GLOB = os.path.expanduser(
    "~/.cache/huggingface/hub/models--ctheodoris--Geneformer/snapshots/*/Geneformer-V2-104M"
)
VALID_ARMS = ("axis_ft", "soft_spearman", "flow", "ce")
# 4-class CE head (the naive arm's head); ordinal labels at loss time.
N_CE_CLASSES = N_ORDINAL
DEFAULT_LAMBDA = {"axis_ft": 0.5, "soft_spearman": 0.5, "flow": 0.5, "ce": 0.0}
# Label space: author31 = the locked 31 author_cell_type classes (default,
# unchanged for every pre-existing invocation); epi4 = epithelial cells only,
# labels collapsed to the 4 evaluation stages (LSC<TAC<basal<differentiated).
LABEL_SPACES = ("author31", "epi4")
DEFAULT_LABEL_SPACE = "author31"
# Canonical stage names of the 4-class label space (== evaluation class set).
EPI4_CLASS_NAMES = {0: "LSC", 1: "TAC", 2: "basal", 3: "differentiated"}


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


def apply_label_space(train_data, id_class_dict, label_space):
    """Restrict the prepared train dataset to ``label_space``.

    Returns ``(train_data, id_class_dict, info)``.

    ``author31`` (default) returns the inputs untouched — the locked pipeline.

    ``epi4`` drops every cell whose author_cell_type has no cardinal stage
    (``class_to_ordinal_map`` -> NON_LAYER, the same NON_LAYER set the ordinal
    head masks) and rewrites ``label`` to the 4-class stage index
    (LSC=0 < TAC=1 < basal=2 < differentiated=3, wing+squamous merged into
    differentiated). The returned class dictionary is the 4-class one, so a
    plain CE head over it trains on exactly the external-evaluation class set.
    """
    if label_space == DEFAULT_LABEL_SPACE:
        return train_data, id_class_dict, {"dropped_non_epithelial": 0}

    class_to_ordinal = class_to_ordinal_map(id_class_dict)
    stage = [class_to_ordinal[int(c)] for c in train_data["label"]]
    keep = [i for i, s in enumerate(stage) if s != NON_LAYER]
    epi = train_data.select(keep).map(
        lambda batch: {
            "label": [class_to_ordinal[int(c)] for c in batch["label"]],
        },
        batched=True,
        desc=f"{label_space} labels",
    )
    id_class_dict_epi4 = {i: EPI4_CLASS_NAMES[i] for i in range(N_ORDINAL)}
    info = {
        "dropped_non_epithelial": len(stage) - len(keep),
        "kept_epithelial": len(keep),
        "stage_counts": {
            EPI4_CLASS_NAMES[i]: int(sum(1 for s in stage if s == i))
            for i in range(N_ORDINAL)
        },
    }
    return epi, id_class_dict_epi4, info


@click.command()
@click.option("--config", default=None, help="YAML config (default: config/default_train_args.yml).")
@click.option("--data_file", default=DEFAULT_DATA_FILE, help="Source h5ad under CORNEAL_FM_DATA.")
@click.option("--epochs", default=None, type=int, help="Training epochs (this study fair protocol = 3).")
@click.option("--batch_size", default=None, type=int, help="Per-device batch size.")
@click.option("--lr", default=None, type=float, help="Learning rate.")
@click.option("--arm", type=click.Choice(list(VALID_ARMS)), required=True,
              help="Loss arm: axis_ft | soft_spearman | flow | ce (plain CE).")
@click.option("--label-space", "label_space", default=DEFAULT_LABEL_SPACE,
              type=click.Choice(list(LABEL_SPACES)),
              help="Label space: author31 (default, the locked 31 author "
                   "classes) | epi4 (epithelial cells only, labels collapsed "
                   "to the 4 stages LSC/TAC/basal/differentiated; requires "
                   "--arm ce).")
@click.option("--lambda-arm", "lambda_arm", default=None, type=float,
              help="Arm loss weight λ (default per arm: 0.5). 0 = plain CE.")
@click.option("--anchor-layer", "anchor_layer", default=None, type=int,
              help="axis_ft only: hidden-state index for the anchor term "
                   "(default = final hidden state = 12 on V2-104M).")
@click.option("--tau-scale", "tau_scale", default=0.1, type=float,
              help="soft_spearman only: soft-rank temperature as a fraction of std.")
@click.option("--k-pairs", "k_pairs", default=256, type=int,
              help="flow only: adjacent-class pairs sampled per batch.")
@click.option("--min-cells-per-class", "min_cells_per_class", default=2, type=int,
              help="flow only: skip adjacent pairs whose class has fewer cells in a batch.")
@click.option("--token_dictionary", default=None, type=click.Path(dir_okay=False),
              help="Geneformer token dictionary pkl (default: from the HF snapshot).")
@click.option("--gene_name_id_dict", default=None, type=click.Path(dir_okay=False),
              help="Geneformer gene-name -> Ensembl pkl (default: from the HF snapshot).")
@click.option("--seed", default=42, type=int, help="Random seed.")
@click.option("--axis-mode", "axis_mode", default=DEFAULT_AXIS_MODE,
              type=click.Choice(list(AXIS_MODES)),
              help="Frozen axis definition: centroid_pca (NEW primary, the "
                   "eval metric's training-side analog) | marker_score "
                   "(original, reference).")
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
    arm: str,
    lambda_arm: float | None,
    anchor_layer: int | None,
    tau_scale: float,
    k_pairs: int,
    min_cells_per_class: int,
    token_dictionary: str | None,
    gene_name_id_dict: str | None,
    seed: int,
    axis_mode: str,
    label_space: str,
    strict_fair: bool,
    split_manifest: str | None,
    prepared_train: str | None,
) -> None:
    """Method-arm fine-tuning: tokenize-with-layer -> prepare_data -> trainer."""
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
    lam = DEFAULT_LAMBDA[arm] if lambda_arm is None else float(lambda_arm)
    if label_space != DEFAULT_LABEL_SPACE and arm != "ce":
        # The epi4 arm is a plain-CE baseline; an auxiliary loss term would
        # conflate "CE supervision" with the arm's own regulariser.
        raise click.UsageError(
            f"--label-space {label_space} is only defined for --arm ce "
            f"(got --arm {arm})"
        )
    if arm == "ce" and lam != 0.0:
        raise click.UsageError(f"--arm ce requires lambda=0 (got {lam})")

    # The default label space keeps the historical directory prefix; a
    # non-default one is tagged into the prefix so runs never collide.
    arm_tag = arm if label_space == DEFAULT_LABEL_SPACE else f"{arm}_{label_space}"
    save_dir = get_save_dir(f"method_{arm_tag}", f"lam{lam}_seed{seed}")
    save_dir.mkdir(parents=True, exist_ok=True)
    _marker(
        save_dir,
        f"START arm={arm} lambda={lam} epochs={epochs} bs={batch_size} lr={lr} "
        f"seed={seed} strict_fair={strict_fair} axis_mode={axis_mode} "
        f"label_space={label_space}",
    )
    archive_script(__file__, save_dir)

    try:
        _run_arm(
            save_dir, data_file, epochs, batch_size, lr, arm, lam, seed,
            anchor_layer=anchor_layer, tau_scale=tau_scale, k_pairs=k_pairs,
            min_cells_per_class=min_cells_per_class,
            token_dictionary=token_dictionary, gene_name_id_dict=gene_name_id_dict,
            split_manifest=split_manifest, manifest=manifest, manifest_hash=manifest_hash,
            prepared_train=prepared_train, strict_fair=strict_fair,
            axis_mode=axis_mode, label_space=label_space,
        )
    except Exception:
        tb = traceback.format_exc()
        with open(save_dir / "ERROR_TRACEBACK.txt", "w") as fh:
            fh.write(f"[{datetime.now().isoformat(timespec='seconds')}]\n{tb}\n")
        _marker(save_dir, "ERROR\n" + tb)
        logger.error("Run failed; traceback -> %s/ERROR_TRACEBACK.txt", save_dir)
        raise


def _run_arm(
    save_dir, data_file, epochs, batch_size, lr, arm, lam, seed,
    *, anchor_layer=None, tau_scale=0.1, k_pairs=256, min_cells_per_class=2,
    token_dictionary=None, gene_name_id_dict=None,
    split_manifest=None, manifest=None, manifest_hash=None, prepared_train=None,
    strict_fair=False, axis_mode=DEFAULT_AXIS_MODE, label_space=DEFAULT_LABEL_SPACE,
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

        # (2) prepare_data (31 author_cell_type classes; the 4-class ordinal
        # labels are derived at loss time via class_to_ordinal_map)
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
    # Same column trimming as caft_gf.py, PLUS an 'idx' column the axis_ft
    # arm needs to index its cached frozen h0 embeddings per batch (the
    # geneformer cell collator preserves extra columns as tensors).
    keep = {"input_ids", "length", "label"}
    drop_cols = [c for c in train_data.column_names if c not in keep]
    if drop_cols:
        train_data = train_data.remove_columns(drop_cols)
    # Label-space restriction (default author31 = no-op). epi4 filters to
    # epithelial cells and rewrites 'label' to the 4 stage classes BEFORE the
    # index column is added, so 'idx' stays contiguous.
    train_data, idc, label_space_info = apply_label_space(train_data, idc, label_space)
    if label_space != DEFAULT_LABEL_SPACE:
        _marker(
            save_dir,
            f"LABEL_SPACE {label_space} kept={label_space_info['kept_epithelial']} "
            f"dropped={label_space_info['dropped_non_epithelial']} "
            f"stages={label_space_info['stage_counts']}",
        )
    if "idx" not in train_data.column_names:
        train_data = train_data.add_column("idx", list(range(len(train_data))))
    _marker(save_dir, f"PREPARED train={len(train_data)} num_classes={len(idc)}")
    logger.info("prepared: train=%d  num_classes=%d", len(train_data), len(idc))

    # (3) frozen differentiation axis (cached; fixed before training)
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model_dir = _locate_pretrained()
    # The plain-CE arm uses no auxiliary term, so no head other than its own
    # label space is ever built/used; every pre-existing arm keeps N_CE_CLASSES.
    n_ce_classes = len(idc) if arm == "ce" else N_CE_CLASSES
    model = load_pretrained_classifier(
        model_dir, num_classes=n_ce_classes, torch_dtype=torch_dtype
    )
    # Move the frozen model to GPU NOW: compute_axis() forwards it over all
    # epithelial training cells before any training starts, and
    # load_pretrained_classifier() leaves it on CPU (from_pretrained default).
    # Without this the ~28k-cell axis forward ran for hours on CPU at ~400%
    # with the GPU idle (axis_ref.py compute_axis -> model.bert).
    if torch.cuda.is_available():
        model = model.to("cuda")
    axis_w = None
    axis = None
    cache_key = None
    if arm == "ce":
        # No continuity/anchor/soft-rank term -> the frozen axis is unused.
        _marker(save_dir, "AXIS skipped (arm=ce has no auxiliary term)")
    else:
        td_file = token_dictionary or default_geneformer_dicts()[0]
        gd_file = gene_name_id_dict or default_geneformer_dicts()[1]
        source_checksum = (manifest or {}).get("input_checksums", {}).get("train_h5ad", "")
        cache_key = axis_cache_key(
            src, model_dir, mode=axis_mode, source_checksum=source_checksum
        )
        class_to_ordinal = class_to_ordinal_map(idc)
        axis = compute_axis(
            model,
            train_data,
            class_to_layer_map(idc),
            td_file, gd_file,
            class_to_ordinal=class_to_ordinal,
            cache_path=AXIS_CACHE_FILE,
            cache_key=cache_key,
            mode=axis_mode,
        )
        axis_w = axis.w
        _marker(
            save_dir,
            f"AXIS mode={axis.mode} method={axis.method} rho_marker={axis.rho_marker:.4f} "
            f"rho_pca={axis.rho_pca:.4f} cache={AXIS_CACHE_FILE}",
        )
        logger.info(
            "axis ready: mode=%s method=%s rho_marker=%.4f",
            axis.mode, axis.method, axis.rho_marker,
        )
    # Return the axis forward's reserved GPU pool to the driver before the
    # training phase (GB10 unified memory: the caching allocator's retained
    # segments + training allocations push the kernel toward its OOM watermark
    # and the driver's IOMMU vmalloc mappings can fail).
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # (4) arm training — same bf16-weights setup as caft_gf.py; only the
    # loss placement differs.
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
    out_dir = str(Path(save_dir) / "model")
    if arm == "ce":
        # Plain CE over the active label space: the SAME CAFTTrainer (and the
        # same collator/bf16/optimiser settings) the naive arm is built from,
        # at lambda=0.0, so no auxiliary term is in the graph.
        trainer = build_caft_trainer(
            model, train_data, idc, caft_lambda=0.0,
            training_args_kwargs=targs, output_directory=out_dir,
        )
    elif arm == "axis_ft":
        trainer = build_axis_ft_trainer(
            model, train_data, idc, axis_w, axis_lambda=lam,
            training_args_kwargs=targs, output_directory=out_dir,
            anchor_layer=anchor_layer,
        )
    elif arm == "soft_spearman":
        trainer = build_ss_trainer(
            model, train_data, idc, axis_w, ss_lambda=lam,
            training_args_kwargs=targs, output_directory=out_dir,
            tau_scale=tau_scale,
        )
    else:  # flow
        trainer = build_flow_trainer(
            model, train_data, idc, axis_w, flow_lambda=lam,
            training_args_kwargs=targs, output_directory=out_dir,
            k_pairs=k_pairs, min_cells_per_class=min_cells_per_class,
        )
    t0 = time.time()
    trainer.train()
    elapsed = (time.time() - t0) / 60
    _marker(save_dir, f"TRAINED elapsed_min={elapsed:.2f}")
    logger.info("%s training done in %.1f min", arm, elapsed)

    # (5) provenance
    with open(Path(save_dir) / "method_arm_args.json", "w") as fh:
        json.dump(
            {
                "arm": arm,
                "lambda": lam,
                "loss": {
                    "axis_ft": "CE + lambda*||P_A(h-h0)||^2 (P_A = w w^T, frozen)",
                    "soft_spearman": "CE + lambda*(1 - rho_soft(proj_A(h), y))",
                    "flow": "CE + lambda*mean adjacent-pair mid-point flow deviation",
                    "ce": "plain CE over the active label space (no auxiliary term)",
                }[arm],
                "axis": {
                    "cache": str(AXIS_CACHE_FILE),
                    "cache_key": cache_key,
                    "mode": axis.mode if axis is not None else None,
                    "method": axis.method if axis is not None else None,
                    "rho_marker": axis.rho_marker if axis is not None else None,
                    "rho_pca": axis.rho_pca if axis is not None else None,
                    "n_cells": axis.n_cells if axis is not None else None,
                    "n_axis_cells": axis.n_axis_cells if axis is not None else None,
                    "marker_status": axis.marker_status if axis is not None else None,
                },
                "arm_hyperparams": {
                    "anchor_layer": anchor_layer,
                    "tau_scale": tau_scale,
                    "k_pairs": k_pairs,
                    "min_cells_per_class": min_cells_per_class,
                },
                "label_space": label_space,
                "label_space_info": label_space_info,
                "class_names": {int(k): str(v) for k, v in idc.items()},
                "n_ce_classes": n_ce_classes,
                "layer_dist": info["layer_dist"],
                "num_classes": len(idc),
                "train_cells": len(train_data),
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "per_device_train_batch_size": PER_DEVICE_TRAIN_BATCH_SIZE,
                "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
                "effective_batch_size_per_device": PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS,
                "seed": seed,
                "strict_fair": strict_fair,
                "split_manifest": split_manifest,
                "manifest_sha256": manifest_hash,
                "prepared_train": prepared_train,
                "elapsed_min": round(elapsed, 2),
                "log_history_tail": trainer.state.log_history[-3:],
            },
            fh, indent=2, default=str,
        )
    with open(Path(save_dir) / "METHOD_ARM_DONE", "w") as fh:
        fh.write(datetime.now().isoformat(timespec="seconds"))
    _marker(save_dir, "DONE")
    logger.info("✅ method arm %s (λ=%.3f) complete -> %s", arm, lam, save_dir)


if __name__ == "__main__":
    main()
