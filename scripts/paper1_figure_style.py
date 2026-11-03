"""Shared visual system for the Paper 1 figure set (Patterns, two-column).

Every rebuilt manuscript figure imports its style, arm vocabulary and colours
from this module so that the whole set reads as one publication.

Retired aliases
---------------
"CORN", "ordinal-naive", "naive FT", "naive" and "Fine-tune lambda = 0" are
retired arm names. They must never appear in any new figure; use the canonical
keys in :data:`ARMS` instead.

The module is intentionally side-effect free at import time. Call
:func:`set_style` before drawing and :func:`save_figure` for every output.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

# ---------------------------------------------------------------------------
# Geometry (Patterns)
# ---------------------------------------------------------------------------

SINGLE_COLUMN_MM = 85.0
DOUBLE_COLUMN_MM = 178.0  # 7.01 in, the two-column text width
MM_PER_INCH = 25.4

# ---------------------------------------------------------------------------
# Locked type scale and structural ink (Patterns house style, 2026-10-19)
# ---------------------------------------------------------------------------
# Four tiers only. Nothing between TYPE["title"] and TYPE["letter"], and
# nothing below TYPE["tick"]. Sizes are points at the final 178 mm canvas.
TYPE = {
    "tick": 7.0,      # tick labels, legend entries
    "axis": 7.5,      # axis titles
    "title": 8.0,     # panel titles, regular weight, sentence case
    "letter": 10.0,   # panel letters, bold
}

# Structural ink. Grey is reserved for the reference/baseline arm role; rules
# and grids use their own neutrals so a rule can never be mistaken for an arm.
RULE_COLOR = "#B0B0B0"   # zero lines, null lines, structural rules
GRID_COLOR = "#E8E8E8"   # gridlines
RULE_LW = 0.6
GRID_LW = 0.4

# Legend keys that encode a marker *shape* rather than an arm print in one
# neutral dark grey, so a shape key can never be read as an arm colour. This is
# deliberately the same value as the ``frozen`` arm: a shape key carries no
# line, so the two keys stay distinguishable by their geometry, not by hue.
SHAPE_KEY_COLOR = "#4D4D4D"

# Line weights and marker sizes (points). PRINT_FLOOR is the thinnest weight
# that survives 300 dpi output.
LW_MAIN = 1.3
LW_SECONDARY = 0.8
LW_SEED = 0.75
LW_REFERENCE = 0.6
PRINT_FLOOR = 0.6
MARKER_PT = 2.64         # 11 px at 300 dpi

# Panel-letter placement: letter top never closer than 2 mm to the trim.
LETTER_TOP_PX = 24       # 2 mm at 300 dpi

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = PROJECT_ROOT / "figures" / "paper1"
MANIFEST_PATH = FIGURE_DIR / "figure_manifest.json"

# ---------------------------------------------------------------------------
# Arm vocabulary
# ---------------------------------------------------------------------------

# Canonical key -> display label. Insertion order is the display order.
ARMS: "OrderedDict[str, str]" = OrderedDict(
    [
        ("frozen", "frozen"),
        ("ce_naive", "CE-naive"),
        ("ordinal", "ordinal head"),
        ("ordinal_cont", "ordinal + continuity"),
        ("lora", "LoRA"),
        ("ordinal_lora", "ordinal + LoRA"),
        ("caft_intended", "CAFT-intended"),
        ("hvg_pca", "HVG/PCA"),
    ]
)

# Okabe-Ito-derived swatches, allocated one per arm. Adjacent arms in ARMS
# order are separated by hue and/or luminance under deuteranopia simulation:
#   frozen #4D4D4D d.grey -> ce_naive #0072B2 blue      (hue)
#   ce_naive blue         -> ordinal #009E73 green      (hue)
#   ordinal green         -> ordinal_cont #56B4E9 sky   (hue/luminance)
#   ordinal_cont sky      -> lora #CC79A7 pink          (luminance)
#   lora pink             -> ordinal_lora #D55E00 verm. (hue/luminance)
#   ordinal_lora verm.    -> caft_intended #E69F00 or.  (luminance)
#   caft_intended orange  -> hvg_pca #949494 grey       (saturation)
# No two arms share a hex value.
#
# Two deviations from the eight base Okabe-Ito swatches are deliberate:
#   * ``frozen`` is dark grey rather than pure black, so an arm mark can never
#     be confused with axis, spine or annotation ink, and it matches the
#     dark-grey frozen convention of the previous figure set.
#   * ``hvg_pca`` is neutral grey rather than Okabe-Ito yellow (#F0E442),
#     which has a contrast ratio near 1.1:1 on white and is unusable for
#     lines, markers or bar fills. HVG/PCA is a reference arm, so a neutral
#     grey also reads correctly as "not a model arm".
ARM_COLORS: "OrderedDict[str, str]" = OrderedDict(
    [
        ("frozen", "#4D4D4D"),        # dark grey (reference convention)
        ("ce_naive", "#0072B2"),      # Okabe-Ito blue
        ("ordinal", "#009E73"),       # Okabe-Ito bluish green
        ("ordinal_cont", "#56B4E9"),  # Okabe-Ito sky blue
        ("lora", "#CC79A7"),          # Okabe-Ito reddish purple
        ("ordinal_lora", "#D55E00"),  # Okabe-Ito vermillion
        ("caft_intended", "#E69F00"),  # Okabe-Ito orange
        ("hvg_pca", "#949494"),       # neutral grey (reference arm)
    ]
)

# The eight Okabe-Ito swatches are all consumed by the eight arms, so the four
# cell-type fills are the remaining reserved family: 40%-white light variants
# of the blue/orange/green/pink swatches. They share no hex with ARM_COLORS, so
# an arm colour can never be mistaken for a cell type. Use these for UMAP/point
# or area fills, not for arm-coded bars.
CELLTYPE_COLORS: "OrderedDict[str, str]" = OrderedDict(
    [
        ("LSC", "#66AAD1"),            # light Okabe-Ito blue
        ("TAC", "#F0C566"),            # light Okabe-Ito orange
        ("basal", "#66C5AB"),          # light Okabe-Ito bluish green
        ("differentiated", "#E0AFCA"),  # light Okabe-Ito reddish purple
    ]
)

# ---------------------------------------------------------------------------
# Heatmap colormaps (plain hex lists, light -> dark / left -> right)
# ---------------------------------------------------------------------------

# Diverging: signed values, Okabe-Ito blue -> neutral -> Okabe-Ito orange.
DIVERGING = [
    "#0072B2",
    "#3E8FC4",
    "#7DB4D8",
    "#BFD9E8",
    "#F2F2F2",
    "#F6DFBC",
    "#F0C566",
    "#E69F00",
    "#B87200",
]

# Sequential: non-negative magnitudes, white -> deep blue.
SEQUENTIAL = [
    "#F7FBFF",
    "#DEEBF7",
    "#C6DBEF",
    "#9ECAE1",
    "#74ADDA",
    "#4A90C4",
    "#2A72AE",
    "#0B5592",
    "#08306B",
]

# Fail fast if the palette and the vocabulary ever drift apart.
if set(ARM_COLORS) != set(ARMS):
    raise RuntimeError("ARM_COLORS keys must match ARMS keys exactly")


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------


def set_style() -> None:
    """Apply the shared rcParams. Idempotent: safe to call repeatedly."""
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial"],
            "font.size": TYPE["tick"],
            "axes.labelsize": TYPE["axis"],
            "axes.titlesize": TYPE["title"],
            "xtick.labelsize": TYPE["tick"],
            "ytick.labelsize": TYPE["tick"],
            "legend.fontsize": TYPE["tick"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.linewidth": 0.5,
            # Ticks: 8 px long, 2 px wide, outward, left/bottom only.
            "xtick.major.width": 0.48,
            "ytick.major.width": 0.48,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 1.92,
            "ytick.major.size": 1.92,
            "xtick.minor.size": 1.0,
            "ytick.minor.size": 1.0,
            # One grid style everywhere, always behind the data.
            "grid.color": GRID_COLOR,
            "grid.linewidth": GRID_LW,
            "axes.grid": False,
            # Legend grammar: unframed, compact, shape keys in reference grey.
            "legend.handlelength": 12.0,
            "legend.handletextpad": 0.6,
            "legend.labelspacing": 0.28,
            "legend.borderpad": 0.2,
            "lines.markersize": MARKER_PT,
            # Legends are never framed in the reference set.
            "legend.frameon": False,
            # Artwork text is black except where colour itself is the datum.
            "text.color": "black",
            "axes.labelcolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
            "savefig.dpi": 600,
            "figure.dpi": 150,
            "figure.constrained_layout.use": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.transparent": False,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def figsize(width_mm: float = DOUBLE_COLUMN_MM, height_mm: float | None = None):
    """Return a matplotlib ``figsize`` tuple in inches.

    Parameters
    ----------
    width_mm
        Figure width in millimetres. Defaults to the Patterns two-column
        width (178 mm); pass ``SINGLE_COLUMN_MM`` for a one-column panel.
    height_mm
        Figure height in millimetres. When omitted, the golden ratio
        (``width_mm / 1.618``) is used.
    """
    if height_mm is None:
        height_mm = width_mm / 1.618033988749895
    return (width_mm / MM_PER_INCH, height_mm / MM_PER_INCH)


def panel_label(ax, letter: str, x: float = -0.14, y: float = 1.06):
    """Place an 8 pt bold panel letter at a consistent axes-relative slot."""
    return ax.text(
        x,
        y,
        letter,
        transform=ax.transAxes,
        # Published Patterns panel letters run ~1.3-1.5x the axis-label height;
        # 7.5 pt axis labels therefore put the letter at ~9.5 pt.
        fontsize=TYPE["letter"],
        fontweight="bold",
        ha="left",
        va="bottom",
        clip_on=False,
    )


def panel_letter_mm(fig, letter: str, x_mm: float, y_mm: float, va: str = "top"):
    """Place a panel letter in a slot measured in millimetres from the trim.

    ``panel_label`` positions a letter *relative to its own axes*, so a letter
    on a short axes sits lower than one on a tall axes in the same figure. This
    helper anchors the letter to the canvas instead: ``x_mm`` is the distance
    from the left trim and ``y_mm`` the distance from the top trim, which lets
    every panel letter in a figure share one baseline and one margin.

    Parameters
    ----------
    fig
        The figure to annotate.
    letter
        Panel letter, drawn at ``TYPE["letter"]`` in bold.
    x_mm, y_mm
        Slot of the letter measured from the top-left trim, in millimetres.
    va
        Vertical anchoring of the slot: ``"top"`` hangs the letter below
        ``y_mm``, ``"bottom"`` stands it on ``y_mm``.
    """
    width_mm, height_mm = (value * MM_PER_INCH for value in fig.get_size_inches())
    return fig.text(
        x_mm / width_mm,
        1.0 - y_mm / height_mm,
        letter,
        fontsize=TYPE["letter"],
        fontweight="bold",
        ha="left",
        va=va,
        clip_on=False,
    )


def check_value(label: str, value: float, expected: float, tol: float = 5e-4) -> None:
    """Assert a plotted number matches the manuscript's stated value.

    Raises
    ------
    AssertionError
        If ``abs(value - expected) > tol``; the message carries both values.
    """
    if abs(float(value) - float(expected)) > tol:
        raise AssertionError(
            f"{label}: plotted value {value!r} != manuscript value {expected!r} "
            f"(tol={tol:g})"
        )


# ---------------------------------------------------------------------------
# Saving + provenance
# ---------------------------------------------------------------------------


def _calling_script() -> str:
    """Absolute path of the script that called ``save_figure``."""
    frame = sys._getframe(2)  # 0 = _calling_script, 1 = save_figure, 2 = caller
    filename = frame.f_globals.get("__file__")
    if filename:
        return str(Path(filename).resolve())
    return str(Path(sys.argv[0]).resolve())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _assert_no_type3(pdf_path: Path) -> None:
    try:
        proc = subprocess.run(
            ["pdffonts", str(pdf_path)],
            capture_output=True,
            text=True,
            check=True,
        )
    except FileNotFoundError as exc:  # pragma: no cover - environment guard
        raise RuntimeError("pdffonts not found; cannot verify font embedding") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"pdffonts failed on {pdf_path} (exit {exc.returncode}): {exc.stderr}"
        ) from exc
    for line in proc.stdout.splitlines():
        if re.search(r"\bType\s+3\b", line):
            raise RuntimeError(f"Type 3 font in {pdf_path}: {line.strip()}")


def _display_path(path: Path) -> str:
    """Repo-relative path when possible, else absolute (portable provenance)."""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def _write_manifest(entry: dict, name: str) -> None:
    """Append ``entry`` to the manifest under a cross-process lock.

    Parallel figure builds run in separate processes; an unlocked
    read-modify-write would lose entries. The lock file is separate from the
    manifest so that the atomic ``os.replace`` below stays valid.
    """
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    lock_path = MANIFEST_PATH.with_suffix(".lock")
    with lock_path.open("a+") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            data: dict = {}
            if MANIFEST_PATH.exists():
                with MANIFEST_PATH.open(encoding="utf-8") as handle:
                    data = json.load(handle)
                if not isinstance(data, dict):
                    raise RuntimeError(f"{MANIFEST_PATH} is not a JSON object")
            data[name] = entry
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=MANIFEST_PATH.parent,
                prefix=".figure_manifest.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                json.dump(data, handle, indent=2)
                handle.write("\n")
                tmp_name = handle.name
            os.replace(tmp_name, MANIFEST_PATH)
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def save_figure(fig, name: str, outdir: str = "figures/paper1", inputs=None):
    """Write ``<name>.pdf`` and ``<name>.png`` and record provenance.

    Parameters
    ----------
    fig
        The matplotlib figure.
    name
        Output stem; one figure has exactly one producer for a given ``name``.
    outdir
        Output directory (created if missing).
    inputs
        Optional mapping ``name -> path`` of every data file the figure read.
        Each path must exist.

    Returns
    -------
    tuple of pathlib.Path
        ``(pdf_path, png_path)``.
    """
    if inputs is None:
        inputs = {}

    out_dir = Path(outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"{name}.pdf"
    png_path = out_dir / f"{name}.png"

    script = _calling_script()
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    metadata = {
        "Title": name,
        "Author": "Corneal-FM Paper 1",
        "Creator": script,
        "Subject": f"generated {timestamp}",
    }

    fig.savefig(
        pdf_path,
        format="pdf",
        bbox_inches=None,
        facecolor="white",
        transparent=False,
        metadata=metadata,
    )
    fig.savefig(
        png_path,
        format="png",
        dpi=600,
        bbox_inches=None,
        facecolor="white",
        transparent=False,
    )

    _assert_no_type3(pdf_path)

    input_hashes = {}
    for input_name, input_path in inputs.items():
        input_file = Path(input_path)
        if not input_file.exists():
            raise FileNotFoundError(
                f"declared input {input_name!r} not found: {input_file.resolve()}"
            )
        input_hashes[input_name] = {
            "path": _display_path(input_file),
            "sha256": _sha256(input_file),
        }

    width_in, height_in = fig.get_size_inches()
    entry = {
        "script": script,
        "timestamp_utc": timestamp,
        "output_pdf": _display_path(pdf_path),
        "output_png": _display_path(png_path),
        "pdf_sha256": _sha256(pdf_path),
        "png_sha256": _sha256(png_path),
        "inputs": input_hashes,
        "width_mm": round(float(width_in) * MM_PER_INCH, 3),
        "height_mm": round(float(height_in) * MM_PER_INCH, 3),
    }
    _write_manifest(entry, name)

    return pdf_path, png_path


if __name__ == "__main__":  # pragma: no cover - manual smoke entry point
    set_style()
    print("ARMS:", list(ARMS))
    print("ARM_COLORS:", dict(ARM_COLORS))
    print("CELLTYPE_COLORS:", dict(CELLTYPE_COLORS))
