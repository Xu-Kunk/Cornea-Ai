#!/usr/bin/env python
"""Corneal-FM model release packager.

Creates a self-contained .zip archive with:
  - best_model.pt          (fine-tuned model weights)
  - vocab.json / gene_vocab.json  (gene vocabulary)
  - config.yml             (training configuration)
  - run.log                (training log, if available)
  - README.md              (usage instructions)

Aligned with eye-scGPT protocol release pattern:
  finetuned_AiO.zip → best_model.pt + vocab.json + train_args.yml + run.log

Usage:
    python protocol_11_release.py --model_dir save/gf_corneal_20260701-120000
    python protocol_11_release.py --model_dir save/scgpt_corneal_20260701-120000 --name corneal_scgpt_v1
"""

from __future__ import annotations

import argparse
import datetime
import json
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAVE_ROOT = PROJECT_ROOT / "save"
RELEASES_ROOT = PROJECT_ROOT / "releases"


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


def find_model_files(model_dir: Path) -> dict[str, Path]:
    """Locate model checkpoint, vocab, and config files in a model directory.

    Parameters
    ----------
    model_dir : Path
        Path to the trained model output directory.

    Returns
    -------
    dict[str, Path]
        Mapping of file type → file path. Keys: ``checkpoint``, ``vocab``,
        ``config``, ``log``.
    """
    files: dict[str, Path] = {}

    # Checkpoint
    ckpt_candidates = list(model_dir.glob("**/*.pt")) + list(model_dir.glob("**/*.pth"))
    if ckpt_candidates:
        # Prefer best_model.pt or geneformer_model.pt / scgpt_model.pt
        named = [p for p in ckpt_candidates if "best" in p.name.lower()]
        gf_named = [p for p in ckpt_candidates if "geneformer" in p.name.lower()]
        scgpt_named = [p for p in ckpt_candidates if "scgpt" in p.name.lower()]
        files["checkpoint"] = (named or gf_named or scgpt_named or ckpt_candidates)[0]

    # Gene vocabulary (JSON)
    vocab_candidates = list(model_dir.glob("**/*vocab*.json")) + list(
        model_dir.glob("**/gene_vocab*.json")
    )
    if vocab_candidates:
        files["vocab"] = vocab_candidates[0]

    # Config
    config_candidates = list(model_dir.glob("**/*.yml")) + list(model_dir.glob("**/*.yaml"))
    config_named = [p for p in config_candidates if "config" in p.name.lower() or "train_args" in p.name.lower()]
    if config_named:
        files["config"] = config_named[0]
    elif config_candidates:
        files["config"] = config_candidates[0]

    # Log
    log_candidates = list(model_dir.glob("**/*.log"))
    if log_candidates:
        files["log"] = log_candidates[0]

    # Copy the calling script if present
    script_candidates = list(model_dir.glob("**/protocol_0*.py"))
    if script_candidates:
        files["script"] = script_candidates[0]

    return files


def build_readme(model_name: str, model_dir: Path, files: dict[str, Path]) -> str:
    """Generate a README.md for the release archive.

    Parameters
    ----------
    model_name : str
        Human-readable model name.
    model_dir : Path
        Source model directory.
    files : dict[str, Path]
        Located model files.

    Returns
    -------
    str
        Markdown README content.
    """
    ckpt_size_mb = files["checkpoint"].stat().st_size / (1024 * 1024) if "checkpoint" in files else 0

    lines = [
        f"# {model_name}",
        "",
        f"> Corneal-FM v7.1 — Tissue-specific single-cell foundation model for corneal research",
        f"> Released: {datetime.date.today().isoformat()}",
        f"> Source: {model_dir.name}",
        "",
        "## Contents",
        "",
        "| File | Description |",
        "|------|-------------|",
    ]

    if "checkpoint" in files:
        lines.append(f"| `{files['checkpoint'].name}` | Model weights ({ckpt_size_mb:.0f} MB) |")
    if "vocab" in files:
        lines.append(f"| `{files['vocab'].name}` | Gene vocabulary mapping |")
    if "config" in files:
        lines.append(f"| `{files['config'].name}` | Training configuration |")
    if "log" in files:
        lines.append(f"| `{files['log'].name}` | Training log |")

    lines += [
        "",
        "## Quick Start",
        "",
        "```python",
        "import torch",
        "from corneal_fm.trainer import GeneformerTrainer, scGPTTrainer",
        "",
        f"# Load model",
        f"trainer = GeneformerTrainer.load('{model_name}')",
        f"# Or: trainer = scGPTTrainer.load('{model_name}')",
        "",
        "# Run inference",
        "embeddings = trainer.model.encode(cells)",
        "```",
        "",
        "## Citation",
        "",
        "If you use Corneal-FM in your research, please cite:",
        "",
        "> [Paper citation — to be added upon publication]",
        "",
        "## License",
        "",
        "MIT License. See LICENSE file for details.",
    ]

    return "\n".join(lines)


def create_release(
    model_dir: Path,
    output_name: str | None = None,
    output_dir: Path | None = None,
) -> Path:
    """Package a trained model into a release .zip archive.

    Parameters
    ----------
    model_dir : Path
        Path to the trained model directory.
    output_name : str | None
        Custom archive name (without .zip). Auto-generated if None.
    output_dir : Path | None
        Output directory. Defaults to ``releases/``.

    Returns
    -------
    Path
        Path to the created .zip file.

    Raises
    ------
    FileNotFoundError
        If no checkpoint file is found in *model_dir*.
    """
    model_dir = model_dir.resolve()
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    files = find_model_files(model_dir)
    if "checkpoint" not in files:
        raise FileNotFoundError(f"No checkpoint (.pt/.pth) found in {model_dir}")

    out_dir = output_dir or RELEASES_ROOT
    out_dir.mkdir(parents=True, exist_ok=True)

    name = output_name or f"corneal_fm_{model_dir.name}"
    zip_path = out_dir / f"{name}.zip"

    model_type = "Geneformer" if "geneformer" in files["checkpoint"].name.lower() else "scGPT"
    readme = build_readme(f"Corneal-FM ({model_type})", model_dir, files)

    print(f"📦 Creating release: {zip_path.name}")
    print(f"   Model:  {files['checkpoint'].name}")
    print(f"   Source: {model_dir}")

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # README
        zf.writestr("README.md", readme)

        # Checkpoint
        arcname = "best_model.pt" if "best" not in files["checkpoint"].name else files["checkpoint"].name
        zf.write(files["checkpoint"], arcname=arcname)

        # Vocab
        if "vocab" in files:
            zf.write(files["vocab"], arcname="gene_vocab.json")

        # Config
        if "config" in files:
            zf.write(files["config"], arcname="train_args.yml")

        # Log
        if "log" in files:
            zf.write(files["log"], arcname="run.log")

        # Script snapshot
        if "script" in files:
            zf.write(files["script"], arcname="training_script.py")

    size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"   Size:   {size_mb:.1f} MB")
    print(f"   Path:   {zip_path}")
    print("✅ Release created successfully.")
    print(f"\n   Usage: unzip {zip_path.name} -d ./{name}/")
    print(f"          python protocol_08_inference_sclong.py --model_dir ./{name}/")

    return zip_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for model release packaging."""
    parser = argparse.ArgumentParser(
        description="Corneal-FM model release packager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python protocol_11_release.py --model_dir save/gf_corneal_20260701-120000
  python protocol_11_release.py --model_dir save/scgpt_corneal_20260701 --name corneal_scgpt_v1
  python protocol_11_release.py --model_dir save/gf_corneal_20260701 --output_dir ./models/
""",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Path to the trained model output directory.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Custom archive name (without .zip). Auto-generated if not provided.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: releases/).",
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    out_dir = Path(args.output_dir) if args.output_dir else None

    create_release(model_dir, args.name, out_dir)


if __name__ == "__main__":
    main()
