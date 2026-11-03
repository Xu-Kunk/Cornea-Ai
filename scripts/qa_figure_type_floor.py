#!/usr/bin/env python3
"""Check the in-figure type floor of every rendered this study figure.

Rationale
---------
The figure set's figures print at 178 mm wide (the target journal two-column
width). Published *the journal* figures set their in-figure text at roughly 7 pt:
measured with ``pdftotext -bbox``, in-figure words in three published the journal
articles have a bounding-box height of 6.28--6.72 pt, while body and caption
text measures 7.62 pt.

A word's bounding-box height is slightly smaller than its nominal font size and
depends on which glyphs it contains, so this check is deliberately about the
*distribution*, not about a single word: a figure whose mode sits at ~7.8 pt is
comfortably above the journal's own floor, while a figure with hundreds of words
below 6 pt has been scaled down past legibility.

Usage
-----
    python3 scripts/qa_figure_type_floor.py figures/paper1/*.pdf
    python3 scripts/qa_figure_type_floor.py --floor 6.0 figures/paper1/*.pdf

Exit code 0 when every figure passes, 1 otherwise.
"""

from __future__ import annotations

import argparse
import collections
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

XHTML = "{http://www.w3.org/1999/xhtml}word"

# Words made only of these glyphs are excluded from the floor test: they carry
# no ascender or descender, so their bounding box is structurally short even at
# a correct type size.
SHORT_GLYPHS = {"ρ", "σ", "=", "≥", "≤", "±", "−", "-", "+", "·", "×", "·"}

# A figure fails when this fraction of measurable words falls below the floor.
MAX_FRACTION_BELOW = 0.05


def word_heights(pdf: Path) -> list[tuple[float, str]]:
    """Return ``(bbox_height_pt, text)`` for every word in the PDF."""
    result = subprocess.run(
        ["pdftotext", "-bbox", str(pdf), "-"],
        capture_output=True,
        text=True,
        check=True,
    )
    root = ET.fromstring(result.stdout)
    heights = []
    for word in root.iter(XHTML):
        height = float(word.get("yMax")) - float(word.get("yMin"))
        heights.append((height, word.text or ""))
    return heights


def measureable(text: str) -> bool:
    """True when a word's height is a fair proxy for its type size."""
    stripped = text.strip()
    if not stripped:
        return False
    return not all(char in SHORT_GLYPHS for char in stripped)


def check(pdf: Path, floor: float) -> tuple[bool, str]:
    heights = word_heights(pdf)
    usable = [(h, t) for h, t in heights if measureable(t)]
    if not usable:
        return False, "no measurable words"
    below = [(h, t) for h, t in usable if h < floor]
    fraction = len(below) / len(usable)
    modes = collections.Counter(round(h, 1) for h, _ in usable).most_common(3)
    verdict = "PASS" if fraction <= MAX_FRACTION_BELOW else "FAIL"
    detail = (
        f"{verdict}  words={len(usable):4d}  below_floor={len(below):4d} "
        f"({fraction * 100:4.1f}%)  min={min(h for h, _ in usable):4.1f}pt  "
        f"modes={modes}"
    )
    if below and verdict == "FAIL":
        sample = ", ".join(repr(t) for _, t in below[:5])
        detail += f"\n      worst examples: {sample}"
    return verdict == "PASS", detail


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdfs", nargs="+", type=Path)
    parser.add_argument(
        "--floor",
        type=float,
        default=6.0,
        help="minimum acceptable bounding-box height in pt (default 6.0, "
        "corresponding to roughly 7 pt type)",
    )
    args = parser.parse_args(argv)

    failed = []
    for pdf in sorted(args.pdfs):
        if not pdf.exists():
            print(f"{pdf}: MISSING")
            failed.append(pdf)
            continue
        ok, detail = check(pdf, args.floor)
        print(f"{pdf.name:44s} {detail}")
        if not ok:
            failed.append(pdf)

    print(f"\n{len(args.pdfs) - len(failed)}/{len(args.pdfs)} figures pass the "
          f"{args.floor} pt floor")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
