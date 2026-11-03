"""Explicit, fail-fast data access for the Paper 1 figure set.

No defaults, no ``or`` fallbacks, no silent substitution: every loader either
returns the real object or raises with the absolute path and, where one exists,
a plausible alternative location. Figure scripts must read every plotted number
through these helpers.

The legacy scripts pointed at ``data/gse153515_annotated.h5ad`` while the file
actually lives at ``data/processed/gse153515_annotated.h5ad``; use
:func:`resolve_data_path` for such source files.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DATA_PROCESSED = DATA_DIR / "processed"
DATA_RAW = DATA_DIR / "raw"
RESULTS = ROOT / "results"
ARTIFACTS = ROOT / "artifacts"
FIGDATA = ROOT / "figures" / "paper1"

_TIER_B_DIR = ARTIFACTS / "paper1_tier_b_eval"


# ---------------------------------------------------------------------------
# Raw loaders
# ---------------------------------------------------------------------------


def _plausible_alternative(path: Path) -> Path | None:
    if path.parent != DATA_PROCESSED and (DATA_PROCESSED / path.name).exists():
        return DATA_PROCESSED / path.name
    return None


def _missing(path: Path, kind: str) -> FileNotFoundError:
    message = f"{kind} not found: {path.resolve()}"
    alternative = _plausible_alternative(path)
    if alternative is not None:
        message += f"; plausible alternative: {alternative.resolve()}"
    return FileNotFoundError(message)


def load_json(path):
    """Load a JSON document, or raise :class:`FileNotFoundError`."""
    path = Path(path)
    if not path.exists():
        raise _missing(path, "JSON file")
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_csv(path):
    """Load a CSV file as a :class:`pandas.DataFrame`, or raise."""
    path = Path(path)
    if not path.exists():
        raise _missing(path, "CSV file")
    return pd.read_csv(path)


def load_npz(path):
    """Load a ``.npz`` archive mapping names to arrays, or raise."""
    path = Path(path)
    if not path.exists():
        raise _missing(path, "NPZ archive")
    return np.load(path)


def resolve_data_path(name):
    """Resolve a bare filename against ``data/processed``, ``data``, ``data/raw``.

    Raises
    ------
    FileNotFoundError
        Listing all three candidate locations.
    """
    candidate = Path(name)
    if candidate.is_absolute():
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"data file not found: {candidate}")
    tried = [DATA_PROCESSED / candidate, DATA_DIR / candidate, DATA_RAW / candidate]
    for path in tried:
        if path.exists():
            return path
    listing = "\n".join(f"  - {path.resolve()}" for path in tried)
    raise FileNotFoundError(f"could not resolve data file {name!r}; tried:\n{listing}")


# ---------------------------------------------------------------------------
# Named accessors (each asserts existence)
# ---------------------------------------------------------------------------


def _require(path: Path, description: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path.resolve()}")
    return path


def percell(name):
    """``artifacts/paper1_tier_b_eval/percell_<name>.npz``."""
    path = _TIER_B_DIR / f"percell_{name}.npz"
    return _require(path, f"percell archive {name!r}")


def tier_b(name):
    """``artifacts/paper1_tier_b_eval/results_<name>.json``."""
    path = _TIER_B_DIR / f"results_{name}.json"
    return _require(path, f"tier-B result {name!r}")


def fig_table(name):
    """``figures/paper1/<name>.csv``."""
    path = FIGDATA / f"{name}.csv"
    return _require(path, f"figure table {name!r}")


def result_json(*parts):
    """``results/<parts...>.json`` (``.json`` appended when absent)."""
    path = RESULTS.joinpath(*parts)
    if path.suffix != ".json":
        path = path.with_name(path.name + ".json")
    return _require(path, f"result JSON {'/'.join(parts)}")


def artifact_json(*parts):
    """``artifacts/<parts...>.json`` (``.json`` appended when absent)."""
    path = ARTIFACTS.joinpath(*parts)
    if path.suffix != ".json":
        path = path.with_name(path.name + ".json")
    return _require(path, f"artifact JSON {'/'.join(parts)}")


# ---------------------------------------------------------------------------
# Numbers the manuscript states (with their source location)
# ---------------------------------------------------------------------------

MANUSCRIPT_VALUES = {
    # paper1.tex, Results "fine-tuning reduced the gradient" (frozen arm row of
    # the external-evaluation table; also Table S3 L12 and Table S5).
    "frozen": {"rho": 0.8064},
    # paper1.tex, same Results paragraph: the three adapted CE-naive runs.
    "ce_naive": {"rho_seeds": (-0.0424, -0.0360, 0.0539)},
    # paper1.tex, external-evaluation table, "Ordinal-only (lambda=0)" row;
    # per-seed values from results/paper1_ordinal_eval.
    "ordinal": {"rho": 0.8330, "rho_seeds": (0.8341, 0.8351, 0.8299)},
    # paper1.tex, external-evaluation table, "Ordinal + continuity" row.
    "ordinal_cont": {"rho": 0.8277},
    # paper1.tex, Results "CellPLM's ordinal arm also retained stage ordering":
    # three-seed mean +/- sample SD on Hu0216 (n = 20,158).
    "cellplm": {
        "ce_rho_mean": -0.0071,
        "ce_rho_sd": 0.0346,
        "ordinal_rho_mean": 0.9180,
        "ordinal_rho_sd": 0.0021,
    },
    # paper1.tex, Results "15-epoch best-validation comparison": scGPT
    # ordinal+LoRA, three seeds 42/123/456 (correlation and macro-F1).
    "scgpt": {
        "ordinal_lora_rho_mean": 0.9241,
        "ordinal_lora_rho_sd": 0.0006,
        "ordinal_lora_macro_f1_mean": 0.9659,
        "ordinal_lora_macro_f1_sd": 0.0024,
    },
    # paper1.tex, Results "fixed-frozen-axis control": frozen scGPT-human
    # self-projection on donor Hu0216.
    "scgpt_human": {"frozen_rho": 0.7887},
    # paper1.tex, Methods: external epithelial evaluation set.
    "gse153515": {"n_cells": 14775, "n_donors": 2},
    # paper1.tex, Methods: van Zyl atlas, Hu0216 locked test donor.
    "vanzyl_hu0216": {"n_cells": 20158},
}
