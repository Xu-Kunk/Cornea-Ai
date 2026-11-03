#!/usr/bin/env python3
"""Figure 1 - RETIRED ODF-flat (FODP) generator and PPTX build driver.

.. warning::

   **DO NOT RUN THIS SCRIPT.**  It is kept only as a historical record of how
   the 2026-10-19/20 revision of Figure 1 was produced.  Since 2026-10-22 the
   figure is hand-maintained:

   * the authoritative input is the editable deck
     ``figures/paper1/design/figure1_final.pptx`` (edited directly, panel E
     deleted, panels A-D rebalanced);
   * the print artefacts are exported from that deck with
     ``soffice --headless --convert-to pdf`` + ``pdftoppm -r 600``;
   * ``figures/paper1/design/figure1_final_layout_spec_2026-10-19.md`` is a
     historical record of the pre-2026-10-22 layout, not a source of truth.

   Re-running this script would overwrite the hand-maintained PPTX with the
   obsolete layout (including the removed panel E) and must not be done.

The historical layout was the layout spec table, transcribed into a flat ODF
presentation, converted to an editable ``.pptx`` with LibreOffice, exported to
the print PDF from the PPTX and rasterised at 600 dpi.

Why FODP
--------
``python-pptx`` is unavailable in this environment.  A flat ODF file is a
single XML document that LibreOffice reads natively and writes to PPTX, where
every rectangle, ellipse, line and text run stays editable.

Coordinates
-----------
All geometry is in millimetres from the top-left trim.  The canvas is
178.0 x 124.0 mm with 4.0 mm margins.  ``x`` is the left edge, or the anchor
point when the layout table's anchor column is ``C`` (centre) or ``R``
(right).

Re-running
----------
    DO NOT RUN.  See the warning above: the PPTX is hand-maintained and this
    generator is retired.  It is kept for provenance only.

Where it was run, it rewrote ``figure1_final.fodp``, ``figure1_final.pptx``,
``figure1_pipeline_overview.pdf`` and ``figure1_pipeline_overview.png``, and
refreshed this figure's entry in ``figures/paper1/figure_manifest.json``.
Every intermediate was generated; nothing was hand-edited.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths and build settings
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
DESIGN_DIR = ROOT / "figures" / "paper1" / "design"
FODP_PATH = DESIGN_DIR / "figure1_final.fodp"
PPTX_PATH = DESIGN_DIR / "figure1_final.pptx"
PDF_PATH = ROOT / "figures" / "paper1" / "figure1_pipeline_overview.pdf"
PNG_PATH = ROOT / "figures" / "paper1" / "figure1_pipeline_overview.png"
SPEC_PATH = DESIGN_DIR / "figure1_final_layout_spec_2026-10-19.md"
MANIFEST_PATH = ROOT / "figures" / "paper1" / "figure_manifest.json"

CANVAS_W_MM = 178.0
CANVAS_H_MM = 124.0
PNG_DPI = 600

# Fixed timestamps keep the generated binaries reproducible.
SOURCE_DATE_EPOCH = "1781740800"  # 2026-06-18T00:00:00Z

# ---------------------------------------------------------------------------
# Ink roles (layout spec, "Ink roles")
# ---------------------------------------------------------------------------

INK = "#1A1A1A"
SUB = "#595959"
STRUCTURE = "#B0B0B0"
HAIRLINE = "#D9D9D9"
TINT = "#F2F2F2"
PILL = "#EDEDED"

FROZEN = "#4D4D4D"
CE_NAIVE = "#0072B2"
ORDINAL = "#009E73"
LORA = "#B4618C"
ORDINAL_LORA = "#D55E00"
HVG_PCA = "#949494"

CELL_COLOURS = ("#66AAD1", "#F0C566", "#66C5AB", "#E0AFCA")

FONT = "Arial"

# ---------------------------------------------------------------------------
# Small XML helpers
# ---------------------------------------------------------------------------


def esc(text: str) -> str:
    """Escape a string for XML text content."""
    return (
        text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
    )


def mm(value: float) -> str:
    """Format a millimetre length for ODF."""
    return f"{value:.3f}mm".replace(".000mm", "mm")


def sha256(path: Path) -> str:
    """Return the SHA-256 of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# ODF document builder
# ---------------------------------------------------------------------------


class Presentation:
    """Build a single-page flat ODF presentation."""

    def __init__(self) -> None:
        self._char_styles: dict[tuple, str] = {}
        self._para_styles: dict[str, str] = {}
        self._graphic_styles: dict[tuple, str] = {}
        self._auto_styles: list[str] = []
        self._hatches: list[str] = []
        self._dashes: list[str] = []
        self._shapes: list[str] = []
        self._style_counter = 0

    # -- style factories ---------------------------------------------------

    def _next(self, prefix: str) -> str:
        self._style_counter += 1
        return f"{prefix}{self._style_counter}"

    def char_style(self, size: float, weight: str, colour: str) -> str:
        """Return the name of a character style for one type role."""
        key = (size, weight, colour)
        if key in self._char_styles:
            return self._char_styles[key]
        name = self._next("c")
        bold = "bold" if weight == "b" else "normal"
        self._auto_styles.append(
            f'<style:style style:name="{name}" style:family="text">'
            f'<style:text-properties fo:font-size="{size:.1f}pt" '
            f'fo:font-weight="{bold}" fo:color="{colour}" '
            f'style:font-name="{FONT}"/></style:style>'
        )
        self._char_styles[key] = name
        return name

    def para_style(self, align: str) -> str:
        """Return the name of a zero-margin, 120 % leading paragraph style."""
        if align in self._para_styles:
            return self._para_styles[align]
        name = self._next("p")
        self._auto_styles.append(
            f'<style:style style:name="{name}" style:family="paragraph">'
            f'<style:paragraph-properties fo:text-align="{align}" '
            f'fo:margin-top="0mm" fo:margin-bottom="0mm" '
            f'fo:line-height="120%"/></style:style>'
        )
        self._para_styles[align] = name
        return name

    def graphic_style(
        self,
        *,
        fill: str | None = None,
        hatch: str | None = None,
        stroke: str | None = None,
        stroke_width: float = 0.6,
        dash: str | None = None,
        vertical: str | None = None,
        solid_white_backing: bool = False,
    ) -> str:
        """Return the name of a graphic style.

        ``vertical`` marks a text frame (``top``/``middle``); text frames are
        always fill- and stroke-free.  Shape styles are keyed on every visual
        property, so identical roles share one style.
        """
        key = (
            fill,
            hatch,
            stroke,
            round(stroke_width, 3),
            dash,
            vertical,
            solid_white_backing,
        )
        if key in self._graphic_styles:
            return self._graphic_styles[key]
        name = self._next("g")
        props = []
        if vertical is not None:
            props.append('draw:fill="none"')
            props.append('draw:stroke="none"')
            props.append(f'draw:textarea-vertical-align="{vertical}"')
            props.append('draw:auto-grow-height="false"')
            props.append('draw:auto-grow-width="false"')
            props.append('fo:padding-top="0mm" fo:padding-bottom="0mm"')
            props.append('fo:padding-left="0mm" fo:padding-right="0mm"')
            props.append('fo:wrap-option="wrap"')
        else:
            if hatch is not None:
                props.append('draw:fill="hatch"')
                props.append(f'draw:fill-hatch-name="{hatch}"')
                props.append(
                    'draw:fill-hatch-solid="true"'
                    if solid_white_backing
                    else 'draw:fill-hatch-solid="false"'
                )
                props.append('draw:fill-color="#FFFFFF"')
            elif fill is not None:
                props.append('draw:fill="solid"')
                props.append(f'draw:fill-color="{fill}"')
            else:
                props.append('draw:fill="none"')
            if stroke is not None:
                props.append('draw:stroke="dash"' if dash is not None else 'draw:stroke="solid"')
                props.append(f'svg:stroke-color="{stroke}"')
                props.append(f'svg:stroke-width="{stroke_width:.2f}pt"')
                props.append('svg:stroke-linecap="butt"')
                props.append('draw:stroke-linejoin="round"')
            else:
                props.append('draw:stroke="none"')
            if dash is not None:
                props.append(f'draw:stroke-dash="{dash}_def"')
        self._auto_styles.append(
            f'<style:style style:name="{name}" style:family="graphic">'
            f"<style:graphic-properties {' '.join(props)}/></style:style>"
        )
        self._graphic_styles[key] = name
        return name

    def hatch(self, name: str, style: str, distance: float, rotation: int) -> str:
        """Define a hatch and return its name."""
        if not any(f'draw:name="{name}"' in item for item in self._hatches):
            self._hatches.append(
                f'<draw:hatch draw:name="{name}" draw:style="{style}" '
                f'draw:color="{STRUCTURE}" draw:distance="{distance}mm" '
                f'draw:rotation="{rotation}"/>'
            )
        return name

    def dash(self, name: str, on_mm: float, off_mm: float) -> str:
        """Define a custom dash pattern and return its style token."""
        if not any(f'draw:name="{name}"' in item for item in self._dashes):
            self._dashes.append(
                f'<draw:stroke-dash draw:name="{name}_def" '
                f'draw:distance="{off_mm}mm" draw:dots1="1" '
                f'draw:dots1-length="{on_mm}mm" draw:dots2="0"/>'
            )
        return name

    # -- geometry ----------------------------------------------------------

    def _frame_prefix(self, name: str, style: str, x: float, y: float, w: float, h: float) -> str:
        return (
            f'<draw:frame draw:name="{esc(name)}" draw:style-name="{style}" '
            f'svg:x="{mm(x)}" svg:y="{mm(y)}" '
            f'svg:width="{mm(w)}" svg:height="{mm(h)}">'
        )

    @staticmethod
    def _unit(value: float, extent: float) -> int:
        """Map a millimetre offset to the shape's 21600-unit space.

        Each axis is normalised by its own extent and the enhanced geometry
        declares ``svg:viewBox="0 0 21600 21600"``.  LibreOffice then exports
        an OOXML path whose declared ``w``/``h`` and point bounds agree, so
        PowerPoint and LibreOffice resolve the geometry identically.
        """
        return int(round(value * 21600.0 / extent))

    def rounded_rect(
        self,
        name: str,
        x: float,
        y: float,
        w: float,
        h: float,
        radius: float,
        *,
        fill: str | None = None,
        stroke: str | None = None,
        stroke_width: float = 0.6,
    ) -> None:
        """Append a rounded rectangle with an explicit corner radius."""
        style = self.graphic_style(fill=fill, stroke=stroke, stroke_width=stroke_width)
        radius = min(radius, w / 2.0, h / 2.0)
        ux, uy = self._unit(radius, w), self._unit(radius, h)
        ux1, uy1 = self._unit(w - radius, w), self._unit(h - radius, h)
        kx = self._unit(radius * 0.5522847498, w)
        ky = self._unit(radius * 0.5522847498, h)
        right, bottom = 21600, 21600
        path = (
            f"M {ux} 0 L {ux1} 0 "
            f"C {ux1 + kx} 0 {right} {ky} {right} {uy} "
            f"L {right} {uy1} "
            f"C {right} {uy1 + ky} {ux1 + kx} {bottom} {ux1} {bottom} "
            f"L {ux} {bottom} "
            f"C {ux - kx} {bottom} 0 {uy1 + ky} 0 {uy1} "
            f"L 0 {uy} "
            f"C 0 {uy - ky} {ux - kx} 0 {ux} 0 Z N"
        )
        self._shapes.append(
            f'<draw:custom-shape draw:name="{esc(name)}" '
            f'draw:style-name="{style}" svg:x="{mm(x)}" svg:y="{mm(y)}" '
            f'svg:width="{mm(w)}" svg:height="{mm(h)}">'
            f'<draw:enhanced-geometry draw:type="non-primitive" '
            f'svg:viewBox="0 0 21600 21600" draw:enhanced-path="{path}"/>'
            f"</draw:custom-shape>"
        )

    def rect(
        self,
        name: str,
        x: float,
        y: float,
        w: float,
        h: float,
        *,
        fill: str | None = None,
        hatch: str | None = None,
        stroke: str | None = None,
        stroke_width: float = 0.6,
    ) -> None:
        """Append a plain rectangle, optionally hatch-filled."""
        style = self.graphic_style(
            fill=fill,
            hatch=hatch,
            stroke=stroke,
            stroke_width=stroke_width,
            solid_white_backing=hatch is not None,
        )
        self._shapes.append(
            f'<draw:rect draw:name="{esc(name)}" draw:style-name="{style}" '
            f'svg:x="{mm(x)}" svg:y="{mm(y)}" '
            f'svg:width="{mm(w)}" svg:height="{mm(h)}"/>'
        )

    def ellipse(
        self,
        name: str,
        x: float,
        y: float,
        w: float,
        h: float,
        *,
        fill: str | None = None,
        stroke: str | None = None,
        stroke_width: float = 0.6,
    ) -> None:
        """Append an ellipse."""
        style = self.graphic_style(fill=fill, stroke=stroke, stroke_width=stroke_width)
        self._shapes.append(
            f'<draw:custom-shape draw:name="{esc(name)}" '
            f'draw:style-name="{style}" svg:x="{mm(x)}" svg:y="{mm(y)}" '
            f'svg:width="{mm(w)}" svg:height="{mm(h)}">'
            f'<draw:enhanced-geometry draw:type="ellipse"/>'
            f"</draw:custom-shape>"
        )

    def glyph(
        self,
        name: str,
        kind: str,
        x: float,
        y: float,
        size: float,
        *,
        fill: str | None = None,
        stroke: str | None = None,
        stroke_width: float = 0.6,
    ) -> None:
        """Append a triangle, square or circle marker."""
        if kind == "circ":
            self.ellipse(
                name,
                x,
                y,
                size,
                size,
                fill=fill,
                stroke=stroke,
                stroke_width=stroke_width,
            )
            return
        style = self.graphic_style(fill=fill, stroke=stroke, stroke_width=stroke_width)
        if kind == "tri":
            path = "M 0 21600 L 10800 0 L 21600 21600 Z N"
        elif kind == "sq":
            path = "M 0 0 L 21600 0 L 21600 21600 L 0 21600 Z N"
        else:  # pragma: no cover - defensive
            raise ValueError(f"unknown glyph {kind!r}")
        self._shapes.append(
            f'<draw:custom-shape draw:name="{esc(name)}" '
            f'draw:style-name="{style}" svg:x="{mm(x)}" svg:y="{mm(y)}" '
            f'svg:width="{mm(size)}" svg:height="{mm(size)}">'
            f'<draw:enhanced-geometry draw:type="non-primitive" '
            f'svg:viewBox="0 0 21600 21600" draw:enhanced-path="{path}"/>'
            f"</draw:custom-shape>"
        )

    def line(
        self,
        name: str,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        *,
        colour: str,
        stroke_width: float,
        dash: str | None = None,
    ) -> None:
        """Append a straight line."""
        style = self.graphic_style(stroke=colour, stroke_width=stroke_width, dash=dash)
        self._shapes.append(
            f'<draw:line draw:name="{esc(name)}" draw:style-name="{style}" '
            f'svg:x1="{mm(x1)}" svg:y1="{mm(y1)}" '
            f'svg:x2="{mm(x2)}" svg:y2="{mm(y2)}"/>'
        )

    def arc(
        self,
        name: str,
        box: tuple[float, float, float, float],
        points: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
        *,
        colour: str,
        stroke_width: float,
    ) -> None:
        """Append an arc through ``points[1]`` with a horizontal apex.

        ``points`` are start, apex and end in canvas millimetres.  The curve is
        two quadratic segments that meet at the apex with a horizontal
        tangent.  Every control point stays inside the element's bounding box,
        so the enhanced-path coordinate space is exactly 21600 x 21600; a
        single quadratic through the apex would need a control point above the
        box, which LibreOffice exports as a taller path and PowerPoint then
        compresses against the shape.
        """
        x0, y0, w, h = box
        (sx, sy), (mx, my), (ex, ey) = points

        def point(px: float, py: float) -> str:
            return f"{self._unit(px - x0, w)} {self._unit(py - y0, h)}"

        def quad_to_cubic(ax: float, ay: float, cx: float, cy: float, bx: float, by: float) -> str:
            """Return the control points of a quadratic as a cubic."""
            c1x, c1y = ax + 2.0 / 3.0 * (cx - ax), ay + 2.0 / 3.0 * (cy - ay)
            c2x, c2y = bx + 2.0 / 3.0 * (cx - bx), by + 2.0 / 3.0 * (cy - by)
            return f"{point(c1x, c1y)} {point(c2x, c2y)} {point(bx, by)}"

        path = (
            f"M {point(sx, sy)} "
            f"C {quad_to_cubic(sx, sy, sx, my, mx, my)} "
            f"C {quad_to_cubic(mx, my, ex, my, ex, ey)} N"
        )
        style = self.graphic_style(stroke=colour, stroke_width=stroke_width)
        self._shapes.append(
            f'<draw:custom-shape draw:name="{esc(name)}" '
            f'draw:style-name="{style}" svg:x="{mm(x0)}" svg:y="{mm(y0)}" '
            f'svg:width="{mm(w)}" svg:height="{mm(h)}">'
            f'<draw:enhanced-geometry draw:type="non-primitive" '
            f'svg:viewBox="0 0 21600 21600" draw:enhanced-path="{path}"/>'
            f"</draw:custom-shape>"
        )

    def text(
        self,
        name: str,
        x: float,
        y: float,
        w: float,
        h: float,
        content: str,
        *,
        size: float,
        weight: str = "r",
        colour: str = INK,
        align: str = "L",
        vpos: str = "T",
    ) -> None:
        """Append a zero-padding text frame.

        ``x`` is the left edge.  Callers pass the left edge explicitly, having
        already resolved any ``C``/``R`` anchor from the layout table.
        """
        align_map = {"L": "left", "C": "center", "R": "right"}
        vertical = "middle" if vpos == "M" else "top"
        gstyle = self.graphic_style(vertical=vertical)
        pstyle = self.para_style(align_map[align])
        cstyle = self.char_style(size, weight, colour)
        self._shapes.append(
            self._frame_prefix(name, gstyle, x, y, w, h)
            + f'<draw:text-box><text:p text:style-name="{pstyle}">'
            f'<text:span text:style-name="{cstyle}">{esc(content)}</text:span>'
            f"</text:p></draw:text-box></draw:frame>"
        )

    # -- document ----------------------------------------------------------

    def document(self) -> str:
        """Serialise the complete flat ODF document."""
        hatch_defs = "".join(self._hatches)
        dash_defs = "".join(self._dashes)
        return (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            "<office:document "
            'xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" '
            'xmlns:style="urn:oasis:names:tc:opendocument:xmlns:style:1.0" '
            'xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0" '
            'xmlns:draw="urn:oasis:names:tc:opendocument:xmlns:drawing:1.0" '
            'xmlns:fo="urn:oasis:names:tc:opendocument:xmlns:xsl-fo-compatible:1.0" '
            'xmlns:xlink="http://www.w3.org/1999/xlink" '
            'xmlns:dc="http://purl.org/dc/elements/1.1/" '
            'xmlns:meta="urn:oasis:names:tc:opendocument:xmlns:meta:1.0" '
            'xmlns:number="urn:oasis:names:tc:opendocument:xmlns:datastyle:1.0" '
            'xmlns:presentation="urn:oasis:names:tc:opendocument:xmlns:presentation:1.0" '
            'xmlns:svg="urn:oasis:names:tc:opendocument:xmlns:svg-compatible:1.0" '
            'xmlns:chart="urn:oasis:names:tc:opendocument:xmlns:chart:1.0" '
            'xmlns:dr3d="urn:oasis:names:tc:opendocument:xmlns:dr3d:1.0" '
            'xmlns:math="http://www.w3.org/1998/Math/MathML" '
            'xmlns:form="urn:oasis:names:tc:opendocument:xmlns:form:1.0" '
            'xmlns:script="urn:oasis:names:tc:opendocument:xmlns:script:1.0" '
            'xmlns:ooo="http://openoffice.org/2004/office" '
            'xmlns:ooow="http://openoffice.org/2004/writer" '
            'xmlns:oooc="http://openoffice.org/2004/calc" '
            'xmlns:dom="http://www.w3.org/2001/xml-events" '
            'xmlns:xforms="http://www.w3.org/2002/xforms" '
            'xmlns:xsd="http://www.w3.org/2001/XMLSchema" '
            'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
            'office:version="1.2" '
            'office:mimetype="application/vnd.oasis.opendocument.presentation">'
            "<office:font-face-decls>"
            f'<style:font-face style:name="{FONT}" svg:font-family="{FONT}" '
            'style:font-family-generic="swiss"/>'
            "</office:font-face-decls>"
            '<office:styles><style:style style:name="standard" '
            'style:family="graphic"/></office:styles>'
            "<office:automatic-styles>"
            '<style:page-layout style:name="PM1">'
            "<style:page-layout-properties "
            f'fo:page-width="{CANVAS_W_MM / 10.0}cm" '
            f'fo:page-height="{CANVAS_H_MM / 10.0}cm" '
            'style:print-orientation="portrait" '
            'fo:margin-top="0cm" fo:margin-bottom="0cm" '
            'fo:margin-left="0cm" fo:margin-right="0cm"/>'
            "</style:page-layout>"
            '<style:style style:name="dp1" style:family="drawing-page">'
            '<style:drawing-page-properties draw:fill="solid" '
            'draw:fill-color="#FFFFFF"/></style:style>'
            f"{''.join(self._auto_styles)}"
            f"{hatch_defs}{dash_defs}"
            "</office:automatic-styles>"
            "<office:master-styles>"
            '<style:master-page style:name="DEFAULT" '
            'style:page-layout-name="PM1"/>'
            "</office:master-styles>"
            "<office:body><office:presentation>"
            '<draw:page draw:name="Figure 1" draw:style-name="dp1" '
            'draw:master-page-name="DEFAULT">'
            f"{''.join(self._shapes)}"
            "</draw:page></office:presentation></office:body>"
            "</office:document>\n"
        )


# ---------------------------------------------------------------------------
# The layout, transcribed from the 2026-10-19 element table
# ---------------------------------------------------------------------------


def build_layout(deck: Presentation) -> None:
    """Add every element of the final layout to ``deck``."""
    h45 = deck.hatch("h45", "single", 0.7, 450)
    h90 = deck.hatch("h90", "single", 0.45, 900)
    h45x135 = deck.hatch("h45x135", "double", 0.9, 450)

    # -- containers --------------------------------------------------------
    for name, x, y, w, h in (
        ("K1", 4.0, 4.0, 41.0, 63.0),
        ("K2", 52.0, 4.0, 122.0, 63.0),
        ("K3", 4.0, 75.0, 115.0, 45.0),
        ("K4", 122.0, 75.0, 52.0, 45.0),
    ):
        deck.rounded_rect(name, x, y, w, h, 2.0, stroke=STRUCTURE, stroke_width=0.6)

    # -- tab bands ---------------------------------------------------------
    for name, x, y, w in (
        ("T1", 4.0, 4.0, 41.0),
        ("T2", 52.0, 4.0, 73.5),
        ("T3", 125.5, 4.0, 48.5),
    ):
        deck.rounded_rect(name, x, y, w, 5.4, 2.0, fill=TINT)

    # -- tab under-rules and the B/C seam ---------------------------------
    for name, x1, x2 in (
        ("L1", 4.0, 45.0),
        ("L2", 52.0, 125.5),
        ("L3", 125.5, 174.0),
    ):
        deck.line(name, x1, 9.4, x2, 9.4, colour=HAIRLINE, stroke_width=0.5)
    deck.line("L4", 125.5, 4.0, 125.5, 34.0, colour=HAIRLINE, stroke_width=0.5)
    deck.line("L5", 125.5, 42.0, 125.5, 67.0, colour=HAIRLINE, stroke_width=0.5)

    # -- panel letters and headings ---------------------------------------
    panels = (
        ("A", 6.0, 4.8, 11.2, 4.6, "DATA & SPLIT"),
        ("B", 54.0, 4.8, 59.2, 4.6, "ARMS & MODELS"),
        ("C", 127.5, 4.8, 132.7, 4.6, "DOMAINS"),
        ("D", 6.0, 77.2, 11.2, 77.0, "READOUTS"),
        ("E", 124.5, 77.2, 129.7, 77.0, "STATISTICAL CONTROL"),
    )
    for letter, dx, dy, tx, ty, heading in panels:
        deck.ellipse(f"{letter}00", dx, dy, 3.8, 3.8, fill=INK)
        deck.text(
            f"{letter}00_t",
            dx,
            dy,
            3.8,
            3.8,
            letter,
            size=10.0,
            weight="b",
            colour="#FFFFFF",
            align="C",
            vpos="M",
        )
        deck.text(
            f"{letter}01",
            tx,
            ty,
            33.0 if letter in {"A", "B"} else (30.0 if letter in {"C", "D"} else 43.0),
            4.2,
            heading,
            size=10.0,
            weight="b",
        )

    # -- A: data and split -------------------------------------------------
    deck.text("A02", 8.0, 15.0, 36.0, 3.4, "CELLxGENE atlas", size=8.0, weight="b")
    deck.text("A03", 8.0, 20.6, 36.0, 3.8, "99,606 train cells", size=9.0, weight="b")
    deck.text("A04", 8.0, 26.2, 36.0, 3.8, "20,394 held-out cells", size=9.0, weight="b")
    deck.text("A05", 8.0, 31.8, 36.0, 3.0, "10 studies · 49 donors", size=7.0, colour=SUB)
    deck.rect("A06", 8.0, 36.8, 26.6, 2.0, fill=STRUCTURE)
    deck.rect("A07", 34.6, 36.8, 5.4, 2.0, hatch=h45, stroke=STRUCTURE, stroke_width=0.25)
    deck.rect("A08", 8.0, 36.8, 32.0, 2.0, stroke=STRUCTURE, stroke_width=0.5)
    deck.text("A09", 15.3, 39.4, 12.0, 3.0, "train", size=7.0, colour=SUB, align="C")
    deck.text("A10", 28.0, 39.4, 12.0, 3.0, "held-out", size=7.0, colour=SUB, align="R")
    deck.line("A11", 8.0, 45.4, 44.0, 45.4, colour=HAIRLINE, stroke_width=0.5)
    deck.rounded_rect("A12", 8.0, 47.2, 26.4, 4.4, 1.2, fill=INK)
    deck.text(
        "A12_t",
        8.0,
        47.2,
        26.4,
        4.4,
        "PRE-REGISTERED",
        size=7.0,
        weight="b",
        colour="#FFFFFF",
        align="C",
        vpos="M",
    )
    deck.text("A13", 8.0, 53.2, 36.0, 3.0, "donor-disjoint · 0 shared", size=7.0, colour=SUB)
    deck.text("A14", 8.0, 56.8, 36.0, 3.0, "Matched comparisons", size=7.0, colour=SUB)
    deck.text("A15", 8.0, 59.8, 36.0, 3.0, "as specified", size=7.0, colour=SUB)

    # -- B: arm matrix -----------------------------------------------------
    deck.text("B02", 94.0, 10.4, 26.0, 3.0, "3 architectures", size=7.0, weight="b", align="C")
    for index, (cx, kind) in enumerate(((93.0, "tri"), (105.0, "sq"), (117.0, "circ")), start=1):
        deck.glyph(f"B03_{index}", kind, cx, 13.8, 4.0, stroke=INK, stroke_width=0.7)
    for name, centre, text in (
        ("B04", 95.0, "Geneformer"),
        ("B05", 107.0, "scGPT"),
        ("B06", 119.0, "CellPLM"),
    ):
        width = 18.0 if name == "B04" else 14.0
        deck.text(
            name,
            centre - width / 2.0,
            17.8,
            width,
            3.0,
            text,
            size=7.0,
            weight="b",
            align="C",
        )
    deck.rounded_rect("B07", 56.0, 21.8, 30.4, 4.4, 1.2, fill=PILL)
    deck.text(
        "B07_t",
        56.0,
        21.8,
        30.4,
        4.4,
        "REFERENCE · 2 arms",
        size=7.0,
        weight="b",
        align="C",
        vpos="M",
    )
    deck.rect("B08", 91.0, 27.0, 31.0, 1.2, fill=HVG_PCA)
    deck.line("B09", 91.0, 28.2, 91.0, 29.6, colour=HVG_PCA, stroke_width=0.9)
    deck.line("B10", 121.1, 28.2, 121.1, 29.6, colour=HVG_PCA, stroke_width=0.9)
    deck.rounded_rect("B11", 56.0, 30.4, 2.8, 2.8, 0.5, fill=FROZEN)
    deck.text("B12", 60.2, 30.0, 30.0, 3.4, "frozen", size=8.0, weight="b")
    for index, (cx, kind) in enumerate(((93.0, "tri"), (105.0, "sq"), (117.0, "circ")), start=1):
        deck.glyph(f"B13_{index}", kind, cx, 29.8, 4.0, stroke=FROZEN, stroke_width=0.7)
    deck.rounded_rect("B14", 56.0, 34.8, 2.8, 2.8, 0.5, fill=HVG_PCA)
    deck.text("B15", 60.2, 34.6, 28.0, 3.4, "HVG/PCA", size=8.0, weight="b")
    deck.text("B16", 60.2, 38.2, 28.0, 3.0, "no backbone", size=7.0, colour=SUB)
    deck.rounded_rect("B17", 56.0, 42.4, 29.4, 4.4, 1.2, fill=PILL)
    deck.text(
        "B17_t",
        56.0,
        42.4,
        29.4,
        4.4,
        "FINE-TUNING · 4 arms",
        size=7.0,
        weight="b",
        align="C",
        vpos="M",
    )

    arms = (
        ("CE-naive", CE_NAIVE),
        ("ordinal head", ORDINAL),
        ("LoRA", LORA),
        ("ordinal + LoRA", ORDINAL_LORA),
    )
    row_y = (47.6, 52.0, 56.4, 60.8)
    label_y = (47.8, 52.2, 56.6, 61.0)
    for index, (label, colour) in enumerate(arms, start=1):
        deck.rounded_rect(f"B18_{index}", 56.0, row_y[index - 1], 2.8, 2.8, 0.5, fill=colour)
        deck.text(
            f"B19_{index}",
            60.2,
            label_y[index - 1],
            28.0,
            3.4,
            label,
            size=8.0,
            weight="b",
        )

    # The arms are a recipe vocabulary, not a filled model x arm grid: the
    # ordinal, LoRA and combined recipes were not run on every architecture.
    deck.text("B21", 91.5, 47.6, 33.0, 3.4, "Model-specific", size=7.0, weight="b")
    deck.text("B22", 91.5, 50.9, 33.0, 3.4, "configurations", size=7.0, weight="b")
    deck.text("B23", 91.5, 55.2, 33.0, 3.0, "not all recipes", size=7.0, colour=SUB)
    deck.text("B24", 91.5, 58.2, 33.0, 3.0, "on all models", size=7.0, colour=SUB)

    # -- C: domains --------------------------------------------------------
    deck.text("C02", 131.0, 17.6, 38.0, 3.0, "three evaluation sets", size=7.0, colour=SUB)
    domain_rows = (
        ("C03", "C04", "C05", 23.2, 24.0, 27.6, "held-out donors", "GSE153515 · same tissue", h45),
        ("C06", "C07", "C08", 38.7, 39.5, 43.1, "post-cutoff", "GSE299152 · after cutoff", h90),
        ("C09", "C10", "C11", 54.2, 55.0, 58.6, "cross-tissue", "epidermis · new tissue", h45x135),
    )
    for token, title, sub, sy, ty, sub_y, title_text, sub_text, hatch in domain_rows:
        deck.rect(token, 131.0, sy, 4.0, 4.0, hatch=hatch, stroke=STRUCTURE, stroke_width=0.25)
        deck.text(title, 137.4, ty, 34.0, 3.4, title_text, size=8.0, weight="b")
        deck.text(sub, 137.4, sub_y, 34.0, 3.0, sub_text, size=7.0, colour=SUB)
    for index, y in enumerate((35.4, 50.6), start=1):
        deck.line(f"C12_{index}", 131.0, y, 170.0, y, colour=HAIRLINE, stroke_width=0.5)

    # -- B x C strap -------------------------------------------------------
    deck.line("S01", 104.0, 68.6, 151.0, 68.6, colour=STRUCTURE, stroke_width=0.6)
    deck.text(
        "S02",
        97.5,
        69.6,
        60.0,
        3.0,
        "Coverage varies by model and domain",
        size=7.0,
        colour=SUB,
        align="C",
    )

    # -- flow arrows (open heads drawn as two segments) --------------------
    def open_arrow(
        name: str,
        x1: float,
        y1: float,
        apex_x: float,
        apex_y: float,
        half_height: float,
        back: float,
        width: float,
    ) -> None:
        deck.line(f"{name}_shaft", x1, y1, apex_x, apex_y, colour=INK, stroke_width=width)
        deck.line(
            f"{name}_head1",
            apex_x - back,
            apex_y - half_height,
            apex_x,
            apex_y,
            colour=INK,
            stroke_width=width,
        )
        deck.line(
            f"{name}_head2",
            apex_x - back,
            apex_y + half_height,
            apex_x,
            apex_y,
            colour=INK,
            stroke_width=width,
        )

    open_arrow("F1", 45.6, 33.7, 51.6, 33.7, 2.0, 1.7, 0.9)
    open_arrow("F2", 120.6, 40.0, 126.4, 40.0, 2.0, 1.7, 0.9)

    # -- D: readouts -------------------------------------------------------
    deck.text(
        "D02", 86.0, 78.0, 29.0, 3.0, "evaluated configurations", size=7.0, colour=SUB, align="R"
    )
    deck.line("D03", 37.5, 84.0, 37.5, 116.0, colour=HAIRLINE, stroke_width=0.5)
    deck.line("D04", 71.5, 84.0, 71.5, 116.0, colour=HAIRLINE, stroke_width=0.5)
    deck.text("D05", 8.0, 84.6, 27.0, 3.4, "Gradient fidelity (ρ)", size=8.0, weight="b")
    deck.text("D06", 8.0, 89.8, 27.0, 3.0, "order preserved →", size=7.0, colour=SUB)
    deck.text("D07", 8.0, 92.8, 27.0, 3.0, "higher ρ is better", size=7.0, colour=SUB)
    deck.ellipse("D08", 8.0, 97.5, 11.5, 11.5, stroke=INK, stroke_width=0.9)
    deck.ellipse("D09", 23.5, 97.5, 11.5, 11.5, stroke=INK, stroke_width=0.9)
    open_arrow("D10", 19.5, 103.25, 23.3, 103.25, 1.0, 1.6, 0.9)
    for index in range(10):
        angle = math.radians(-90.0 + 36.0 * index)
        cx = 13.75 + 5.75 * math.cos(angle)
        cy = 103.25 + 5.75 * math.sin(angle)
        deck.ellipse(
            f"D11_{index + 1}",
            cx - 0.85,
            cy - 0.85,
            1.7,
            1.7,
            fill=CELL_COLOURS[index % 4],
        )
    deck.text("D12", 6.75, 110.4, 14.0, 3.0, "reference", size=7.0, colour=SUB, align="C")
    deck.text("D13", 22.25, 110.4, 14.0, 3.0, "fine-tuned", size=7.0, colour=SUB, align="C")

    deck.text("D14", 38.0, 84.6, 33.0, 3.6, "Layer adjacency", size=8.0, weight="b")
    deck.text("D15", 38.0, 88.2, 33.0, 3.6, "accuracy (LAA)", size=8.0, weight="b")
    deck.text("D16_1", 38.0, 92.4, 33.0, 3.0, "Nearest neighbors", size=7.0, colour=SUB)
    deck.text("D16_2", 38.0, 95.4, 33.0, 3.0, "same or adjacent stage", size=7.0, colour=SUB)
    # LAA is measured inside one evaluated population: a cell and its nearest
    # other cell in Euclidean embedding space, with an identical or adjacent
    # stage.  Two neutral markers and a plain link, so the card shows neither a
    # comparison between data domains nor a direction of effect.
    deck.glyph("D17", "circ", 45.2, 101.6, 3.6, fill=INK, stroke=INK, stroke_width=0.6)
    deck.glyph("D18", "circ", 57.2, 101.6, 3.6, fill=INK, stroke=INK, stroke_width=0.6)
    deck.line("D19", 48.8, 103.4, 57.2, 103.4, colour=INK, stroke_width=0.6)
    deck.text("D20_1", 41.0, 110.4, 12.0, 3.0, "cell", size=7.0, colour=SUB, align="C")
    deck.text("D20_2", 51.5, 110.4, 15.0, 3.0, "nearest other", size=7.0, colour=SUB, align="C")

    deck.text("D21", 75.0, 84.6, 39.0, 3.4, "Classification", size=8.0, weight="b")
    deck.text("D22", 75.0, 89.8, 39.0, 3.0, "labels recovered →", size=7.0, colour=SUB)
    deck.text("D23", 75.0, 92.8, 39.0, 3.0, "macro-F1", size=7.0, colour=SUB)
    cloud = ("tri", "sq", "circ")
    for row, (count, x0, y) in enumerate(((4, 74.6, 100.3), (3, 75.65, 102.3), (3, 76.7, 104.3))):
        for column in range(count):
            deck.glyph(
                f"D24_{row + 1}_{column + 1}",
                cloud[(row + column) % 3],
                x0 + 2.1 * column,
                y,
                1.7,
                stroke=INK,
                stroke_width=0.6,
            )
    open_arrow("D25", 83.2, 103.2, 87.2, 103.2, 1.0, 1.6, 0.9)
    for cluster, (x0, kind) in enumerate(((91.2, "tri"), (98.4, "sq"), (105.6, "circ")), start=1):
        for column in range(3):
            deck.glyph(
                f"D26_{cluster}_{column + 1}",
                kind,
                x0 + 2.1 * column,
                102.3,
                1.8,
                fill=INK,
                stroke=INK,
                stroke_width=0.6,
            )
    deck.text("D27", 71.6, 110.4, 14.0, 3.0, "collapsed", size=7.0, colour=SUB, align="C")
    deck.text("D28", 91.4, 110.4, 14.0, 3.0, "rescued", size=7.0, colour=SUB, align="C")

    # -- E: statistical control -------------------------------------------
    # A schematic of the association test only: one observed statistic against
    # the label-shuffle null.  No arm contrast, no p/q values, no arm colours.
    deck.text(
        "E02",
        126.5,
        83.0,
        43.0,
        3.0,
        "positive stage-rank association",
        size=7.0,
        weight="b",
        align="C",
    )
    deck.text(
        "E03",
        126.5,
        86.6,
        43.0,
        3.0,
        "not an arm-versus-arm test",
        size=7.0,
        colour=SUB,
        align="C",
    )
    deck.arc(
        "E04",
        (128.0, 96.5, 40.0, 7.5),
        ((128.0, 104.0), (148.5, 96.5), (168.0, 104.0)),
        colour=STRUCTURE,
        stroke_width=0.6,
    )
    deck.line("E05", 126.5, 104.0, 169.5, 104.0, colour=STRUCTURE, stroke_width=0.5)
    deck.glyph("E06", "circ", 159.5, 101.4, 2.0, fill=INK, stroke=INK, stroke_width=0.4)
    deck.text("E07", 126.5, 106.0, 30.0, 3.0, "label-shuffle null", size=7.0, colour=SUB)
    deck.text("E08", 154.5, 106.0, 12.0, 3.0, "observed", size=7.0, colour=SUB, align="C")
    deck.line("E09", 126.5, 114.0, 169.5, 114.0, colour=HAIRLINE, stroke_width=0.5)
    deck.text(
        "E10",
        126.5,
        115.0,
        43.0,
        3.0,
        "BH–FDR · specified families",
        size=7.0,
        colour=SUB,
    )


# ---------------------------------------------------------------------------
# Build driver
# ---------------------------------------------------------------------------


def run(command: list[str], *, cwd: Path | None = None) -> None:
    """Run a subprocess, raising with its stderr on failure."""
    env = dict(os.environ)
    env["SOURCE_DATE_EPOCH"] = SOURCE_DATE_EPOCH
    result = subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(command)}\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )


def normalise_pptx(path: Path) -> None:
    """Rewrite the PPTX zip with fixed entry timestamps.

    LibreOffice stamps ZIP members with the wall-clock time, so the file hash
    changes on every run even when the geometry is identical.  Rewriting the
    archive with a fixed timestamp makes the byte stream reproducible.
    """
    import zipfile

    with zipfile.ZipFile(path) as source:
        entries = [(info, source.read(info.filename)) for info in source.infolist()]
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as target:
        for info, payload in entries:
            info.date_time = (1980, 1, 1, 0, 0, 0)
            info.create_system = 0
            info.external_attr = 0
            target.writestr(info, payload)


def normalise_pdf(path: Path) -> None:
    """Replace the PDF creation date and document ID with fixed values.

    Both live in the trailer as plain text and the replacements keep the exact
    byte length, so no cross-reference offset moves.
    """
    data = path.read_bytes()
    data = re.sub(rb"D:\d{14}[+-]\d{2}'\d{2}'", b"D:20260618000000+00'00'", data)
    data = re.sub(
        rb"/ID\s*\[\s*<[0-9A-Fa-f]{32}>\s*<[0-9A-Fa-f]{32}>\s*\]",
        b"/ID [ <0123456789ABCDEF0123456789ABCDEF><FEDCBA9876543210FEDCBA9876543210> ]",
        data,
    )
    data = re.sub(
        rb"/DocChecksum\s*/[0-9A-Fa-f]{32}",
        b"/DocChecksum /00000000000000000000000000000000",
        data,
    )
    path.write_bytes(data)


def convert(src: Path, fmt: str, outdir: Path, profile: Path) -> Path:
    """Convert one document with a private LibreOffice profile."""
    run(
        [
            "soffice",
            "--headless",
            "--norestore",
            "--nolockcheck",
            f"-env:UserInstallation=file://{profile}",
            "--convert-to",
            fmt,
            "--outdir",
            str(outdir),
            str(src),
        ]
    )
    produced = outdir / f"{src.stem}.{fmt.split(':')[0]}"
    if not produced.exists():
        raise RuntimeError(f"LibreOffice did not produce {produced}")
    return produced


def build() -> dict[str, str]:
    """Generate the FODP, PPTX, PDF and PNG; return hashes."""
    deck = Presentation()
    build_layout(deck)
    FODP_PATH.parent.mkdir(parents=True, exist_ok=True)
    FODP_PATH.write_text(deck.document(), encoding="utf-8")

    with tempfile.TemporaryDirectory(prefix="figure1-build-") as tmp_name:
        tmp = Path(tmp_name)
        profile = tmp / "lo-profile"
        profile.mkdir()

        pptx_tmp = convert(FODP_PATH, "pptx", tmp, profile)
        shutil.copyfile(pptx_tmp, PPTX_PATH)
        normalise_pptx(PPTX_PATH)

        pdf_tmp = convert(PPTX_PATH, "pdf", tmp, profile)
        shutil.copyfile(pdf_tmp, PDF_PATH)
        normalise_pdf(PDF_PATH)

    run(
        [
            "pdftoppm",
            "-r",
            str(PNG_DPI),
            "-png",
            "-singlefile",
            str(PDF_PATH),
            str(PNG_PATH.with_suffix("")),
        ]
    )

    return {
        "fodp": sha256(FODP_PATH),
        "pptx": sha256(PPTX_PATH),
        "pdf": sha256(PDF_PATH),
        "png": sha256(PNG_PATH),
    }


def update_manifest(hashes: dict[str, str]) -> None:
    """Refresh this figure's manifest entry, preserving every other entry."""
    if MANIFEST_PATH.exists():
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    else:  # pragma: no cover - the figure set ships a populated manifest
        manifest = {}

    previous = manifest.get("figure1_pipeline_overview", {})
    entry = {
        "script": str(Path(__file__).resolve()),
        "producer": "scripts/figure1_layout_pptx.py",
        "producer_kind": "fodp-pptx-generator",
        "timestamp_utc": datetime.now(UTC)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "spec": "figures/paper1/design/figure1_final_layout_spec_2026-10-19.md",
        "spec_sha256": sha256(SPEC_PATH),
        "source_fodp": "figures/paper1/design/figure1_final.fodp",
        "fodp_sha256": hashes["fodp"],
        "output_pptx": "figures/paper1/design/figure1_final.pptx",
        "pptx_sha256": hashes["pptx"],
        "output_pdf": "figures/paper1/figure1_pipeline_overview.pdf",
        "pdf_sha256": hashes["pdf"],
        "output_png": "figures/paper1/figure1_pipeline_overview.png",
        "png_sha256": hashes["png"],
        "inputs": {
            "layout_spec": {
                "path": "figures/paper1/design/figure1_final_layout_spec_2026-10-19.md",
                "sha256": sha256(SPEC_PATH),
            }
        },
        "width_mm": CANVAS_W_MM,
        "height_mm": CANVAS_H_MM,
    }
    # Keys this producer does not own (``superseded_producer`` names the
    # retired matplotlib script) describe the artefact set and must survive a
    # rebuild; without this they are dropped silently on every run.
    for key, value in previous.items():
        entry.setdefault(key, value)
    manifest["figure1_pipeline_overview"] = entry
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    """Build the artefact set and report the sizes."""
    hashes = build()
    update_manifest(hashes)
    print("generated:")
    for path in (FODP_PATH, PPTX_PATH, PDF_PATH, PNG_PATH):
        print(f"  {path.relative_to(ROOT)}  {path.stat().st_size:,} bytes")
    print("sha256:")
    for key, value in hashes.items():
        print(f"  {key:5s} {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
