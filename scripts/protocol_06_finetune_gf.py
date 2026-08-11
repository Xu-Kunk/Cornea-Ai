#!/usr/bin/env python
"""Corneal-FM: Geneformer V2-104M Classifier fine-tuning on corneal cells.

Standard baseline fine-tuning using the current Geneformer Classifier API
(prepare_data -> train_all_data). This script fine-tunes a cell-type classifier
with the usual cross-entropy objective; the CAFT continuity loss is added as a
separate option in a follow-up.

Notes
-----
- Targets NVIDIA GB10 (CUDA 13 sm_121, 128GB unified memory): fp16 + small
  footprint; train with swap off and vLLM stopped.
- The Geneformer pretrained weights (Geneformer-V2-104M) are located under the
  HuggingFace hub cache (override with --pretrained_dir).
- dry_run runs prepare_data (labeling/downsample/split) and stops before training.

Usage
-----
    python scripts/protocol_06_finetune_gf.py --help
    python scripts/protocol_06_finetune_gf.py --max_ncells 200 --epochs 1   # smoke test
    python scripts/protocol_06_finetune_gf.py --epochs 10 --batch_size 32 --lr 5e-5 --seed 42
    python scripts/protocol_06_finetune_gf.py --dry_run                     # validate data prep
    python scripts/protocol_06_finetune_gf.py --freeze_layers 4
"""

from __future__ import annotations

import glob
import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import click
import torch

from corneal_fm.config import archive_script, get_save_dir, load_config, set_seed
from corneal_fm.utils import check_gpu_memory, get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("CORNEAL_FM_DATA", str(PROJECT_ROOT / "data")))
TOKENIZED_DIR = DATA / "gf_tokenized_train"
DATASET_PATH = TOKENIZED_DIR / "corneal_100k.dataset"
DEFAULT_CONFIG = "config/default_train_args.yml"
_HF_SNAP_GLOB = os.path.expanduser(
    "~/.cache/huggingface/hub/models--ctheodoris--Geneformer/snapshots/*/Geneformer-V2-104M"
)


def _locate_pretrained(override: str | None) -> str:
    if override:
        return override
    hits = glob.glob(_HF_SNAP_GLOB)
    if not hits:
        logger.error(
            "Geneformer-V2-104M weights not found under the HuggingFace cache. "
            "Download ctheodoris/Geneformer (hf-mirror.com) or pass --pretrained_dir."
        )
        sys.exit(1)
    return hits[0]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
@click.command()
@click.option("--config", default=None, help="Path to YAML config (default: config/default_train_args.yml).")
@click.option("--epochs", default=None, type=int, help="Number of training epochs.")
@click.option("--batch_size", default=None, type=int, help="Per-device training batch size.")
@click.option("--lr", default=None, type=float, help="Peak learning rate.")
@click.option("--freeze_layers", default=None, type=int, help="Number of Geneformer encoder layers to freeze (0 = train all).")
@click.option("--max_ncells", default=0, type=int, help="Subsample to at most N cells for training (0 = use all).")
@click.option("--test_size", default=0.2, type=float, help="Held-out test fraction prepared by prepare_data (0 = none).")
@click.option("--pretrained_dir", default=None, help="Explicit Geneformer-V2-104M dir (default: auto from HF cache).")
@click.option("--dry_run", is_flag=True, help="Run data preparation only; stop before training.")
@click.option("--no_amp", is_flag=True, help="Disable fp16 mixed precision.")
@click.option("--seed", default=42, type=int, help="Random seed for reproducibility.")
def main(
    config: str | None,
    epochs: int | None,
    batch_size: int | None,
    lr: float | None,
    freeze_layers: int | None,
    max_ncells: int,
    test_size: float,
    pretrained_dir: str | None,
    dry_run: bool,
    no_amp: bool,
    seed: int,
) -> None:
    """Fine-tune Geneformer V2-104M cell-type classifier on corneal scRNA-seq.

    Loads the tokenized corpus produced by protocol_04_tokenize_gf.py, prepares
    labeled data via Classifier.prepare_data, and fine-tunes the pretrained
    Geneformer-V2-104M encoder with a classification head (standard
    cross-entropy). Results are saved under save/ with the script archived.
    """
    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------
    cfg = load_config(config or DEFAULT_CONFIG)
    mp = cfg["model_parameters"]
    mp.setdefault("freeze_layers", 0)
    mp.setdefault("epochs", 10)
    mp.setdefault("batch_size", 32)
    mp.setdefault("learning_rate", 5e-5)
    mp.setdefault("amp", True)
    if epochs is not None:
        mp["epochs"] = epochs
    if batch_size is not None:
        mp["batch_size"] = batch_size
    if lr is not None:
        mp["learning_rate"] = lr
    if freeze_layers is not None:
        mp["freeze_layers"] = freeze_layers
    if no_amp:
        mp["amp"] = False

    set_seed(seed)

    # ------------------------------------------------------------------
    # Save directory & script archive
    # ------------------------------------------------------------------
    save_dir = get_save_dir("gf_v2-104M", "corneal_100k")
    save_dir.mkdir(parents=True, exist_ok=True)
    archive_script(__file__, save_dir)
    prepared_dir = save_dir / "prepared"
    prepared_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Device GPU mem: %s", check_gpu_memory())
    logger.info("Save dir: %s", save_dir)

    if not TOKENIZED_DIR.exists():
        logger.error("Tokenized data directory not found: %s", TOKENIZED_DIR)
        logger.error("Run protocol_04_tokenize_gf.py first to generate the dataset.")
        sys.exit(1)

    model_dir = _locate_pretrained(pretrained_dir)
    logger.info("Pretrained Geneformer-V2-104M: %s", model_dir)
    logger.info(
        "Config: epochs=%d  bs=%d  lr=%.2e  freeze=%d  amp=%s  max_ncells=%s  test_size=%.2f",
        mp["epochs"], mp["batch_size"], mp["learning_rate"],
        mp["freeze_layers"], mp["amp"], max_ncells or "all", test_size,
    )

    # ------------------------------------------------------------------
    # Build classifier + prepare labeled data (new Geneformer API)
    # ------------------------------------------------------------------
    from geneformer import Classifier

    # Column handling (remove_unused_columns / label names / length_column_name)
    # is owned by geneformer's collator + default args. Forcing them here makes
    # the HF Trainer pass the 'length' column into BertForSequenceClassification
    # forward and breaks the geneformer collator, so only hyperparameters are set.
    training_args = {
        "num_train_epochs": int(mp["epochs"]),
        "per_device_train_batch_size": int(mp["batch_size"]),
        "learning_rate": float(mp["learning_rate"]),
        "fp16": bool(mp["amp"]) and torch.cuda.is_available(),
        "max_grad_norm": 1.0,
    }
    clf = Classifier(
        classifier="cell",
        cell_state_dict={"state_key": "cell_type", "states": "all"},
        model_version="V2",
        freeze_layers=mp["freeze_layers"],
        num_crossval_splits=1,
        nproc=4,
        max_ncells=max_ncells if max_ncells else None,
        training_args=training_args,
    )

    prefix = "corneal"
    logger.info("Preparing labeled data from %s ...", DATASET_PATH)
    t_prep = datetime.now()
    clf.prepare_data(
        input_data_file=str(DATASET_PATH),
        output_directory=str(prepared_dir),
        output_prefix=prefix,
        test_size=test_size,
    )
    train_ds = prepared_dir / f"{prefix}_labeled_train.dataset"
    id_class_dict = prepared_dir / f"{prefix}_id_class_dict.pkl"
    if not train_ds.exists():
        train_ds = prepared_dir / f"{prefix}_labeled.dataset"
    logger.info("Prepared data in %.1f s -> %s", (datetime.now() - t_prep).total_seconds(), train_ds)
    logger.info("id_class_dict: %s", id_class_dict)

    if dry_run:
        logger.info("DRY-RUN complete — data prepared OK (no training).")
        return

    # ------------------------------------------------------------------
    # Train (standard cross-entropy classifier head)
    # ------------------------------------------------------------------
    logger.info("Starting Geneformer fine-tuning (standard classifier head) ...")
    t0 = datetime.now()
    try:
        clf.train_all_data(
            model_directory=model_dir,
            prepared_input_data_file=str(train_ds),
            id_class_dict_file=str(id_class_dict),
            output_directory=str(save_dir),
            output_prefix="gf_v2-104M",
            save_eval_output=False,
        )
    except Exception as exc:
        logger.error("Training failed: %s", exc)
        logger.error("Check GPU memory / data compatibility.")
        raise
    logger.info("Training completed in %.1f min", (datetime.now() - t0).total_seconds() / 60)

    # ------------------------------------------------------------------
    # Provenance
    # ------------------------------------------------------------------
    with open(save_dir / "training_args.json", "w") as fh:
        json.dump(training_args, fh, indent=2, default=str)
    logger.info("=" * 60)
    logger.info("✅ Geneformer fine-tuning complete — output at %s", save_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
