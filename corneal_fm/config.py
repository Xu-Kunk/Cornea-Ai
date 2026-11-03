"""Configuration management for Corneal-FM: YAML config, save directories, seeding."""

from __future__ import annotations

import datetime
import random
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG: dict[str, Any] = {
    # Model
    "model_name": "gf-12L-95M",  # Geneformer 12-layer 95M, or scGPT-8L-94M
    "model_type": "geneformer",  # "geneformer" | "scgpt"
    "pretrained_path": None,  # Override default pretrained model path
    # Fine-tuning
    "epochs": 25,
    "batch_size": 24,
    "lr": 5e-5,
    "weight_decay": 0.01,
    "warmup_steps": 1000,
    "freeze_layers": 0,  # Number of transformer layers to freeze (0 = train all)
    "gradient_clip_norm": 1.0,
    "amp": True,  # Automatic mixed precision
    "dry_run": False,  # Dry-run mode to find max batch size
    # Data
    "dataset": "corneal",
    "max_cells": None,  # Limit cells for debugging (None = all)
    "train_split": 0.8,
    "val_split": 0.1,
    "test_split": 0.1,
    # Geneformer-specific
    "geneformer_num_classes": None,  # Auto-detected if None
    "geneformer_filter_data": True,  # Filter data to geneformer vocabulary
    # scGPT-specific
    "scgpt_mask_ratio": 0.15,
    "scgpt_recon_loss": "mse",  # Reconstruction loss: "mse" | "kl"
    "scgpt_nlayers": 4,  # Number of fine-tuning layers
    # Evaluation
    "eval_classification": True,
    "eval_perturbation": False,
    "eval_cross_species": False,
    "eval_embedding": True,
    # System
    "seed": 42,
    "num_workers": 4,
    "save_dir": None,  # Auto-generated if None
    "device": "auto",  # "auto" | "cuda" | "cpu"
}


# ---------------------------------------------------------------------------
# Config loading / saving
# ---------------------------------------------------------------------------


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML config file and merge it with the default configuration.

    Values in the YAML file override the defaults.  The returned dict is
    guaranteed to contain every key from ``_DEFAULT_CONFIG``.

    Parameters
    ----------
    config_path : str | Path
        Path to a ``.yaml`` or ``.yml`` configuration file.

    Returns
    -------
    dict[str, Any]
        Merged configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If *config_path* does not exist.
    yaml.YAMLError
        If the file contains invalid YAML.
    """
    config_path = Path(config_path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, encoding="utf-8") as fh:
        overrides: dict[str, Any] = yaml.safe_load(fh) or {}

    config = _DEFAULT_CONFIG.copy()
    config.update(overrides)
    return config


def save_config(config: dict[str, Any], save_path: str | Path) -> None:
    """Save a configuration dictionary to a YAML file.

    Parameters
    ----------
    config : dict[str, Any]
        Configuration dictionary.
    save_path : str | Path
        Destination path for the YAML file.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as fh:
        yaml.dump(
            config,
            fh,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )


# ---------------------------------------------------------------------------
# Save directory
# ---------------------------------------------------------------------------


def get_save_dir(model_name: str, dataset: str) -> Path:
    """Generate a timestamped save directory.

    The directory path follows the pattern used by eye-scGPT::

        save/{model_name}_{dataset}_{YYYYMMDD-HHMMSS}/

    Parameters
    ----------
    model_name : str
        Model identifier (e.g. ``"gf_v2-104M"``).
    dataset : str
        Dataset name (e.g. ``"corneal"``).

    Returns
    -------
    Path
        Created directory path.
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    save_dir = Path("save") / f"{model_name}_{dataset}_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=True)
    return save_dir


def archive_script(script_path: str | Path, save_dir: str | Path) -> None:
    """Copy the calling script into the save directory.

    This mirrors the eye-scGPT pattern ``cp {__file__} {save_dir}``,
    preserving a snapshot of the training script alongside the results.

    Parameters
    ----------
    script_path : str | Path
        Path to the script to archive (e.g. ``__file__`` from the caller).
    save_dir : str | Path
        Destination directory.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    script_path = Path(script_path)
    if script_path.is_file():
        shutil.copy2(script_path, save_dir)
    else:
        # Fall back to current module if the path is a module reference
        main_module = sys.modules.get("__main__")
        if main_module and hasattr(main_module, "__file__") and main_module.__file__:
            m_path = Path(main_module.__file__)
            if m_path.is_file():
                shutil.copy2(m_path, save_dir)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility across torch, numpy, and Python.

    Parameters
    ----------
    seed : int
        Random seed value (default 42).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Enable deterministic algorithms for PyTorch ops where available.
    # This may slow down training slightly.
    torch.backends.cudnn.deterministic = True  # type: ignore[no-any-unimported]
    torch.backends.cudnn.benchmark = False  # type: ignore[no-any-unimported]
