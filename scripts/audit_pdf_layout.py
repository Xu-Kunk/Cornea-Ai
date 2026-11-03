"""Audit PDF figure/table layout for the Cell Press preprint.

Outputs a structured layout report to stdout and a JSON file:
    artifacts/paper1_preprint/pdf_layout_audit.json
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

# PyMuPDF (pymupdf) exposes the legacy `fitz` namespace. Prefer the modern
# import to silence deprecation warnings.
try:
    import pymupdf as fitz  # type: ignore[import]
except ImportError:
    import fitz  # type: ignore[import]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
PDF_PATH = ROOT / "artifacts" / "paper1_preprint" / "paper1_v17_preprint.pdf"
OUT_JSON = ROOT / "artifacts" / "paper1_preprint" / "pdf_layout_audit.json"

PAGE_TEXT_WIDTH_IN = 7.0  # Cell Press body text width (inches)
MIN_IMG_WIDTH_IN = 3.0
MAX_PAGE_AREA_COVERAGE = 0.80
MIN_PAGE_COVERAGE = 0.30
DRAWING_AREA_THRESHOLD_PT2 = 2000.0


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class ImageRecord:
    page: int
    bbox: tuple[float, float, float, float]
    width_in: float
    height_in: float
    area_ratio: float
    width_ratio: float
    issue: str


@dataclass
class CaptionRecord:
    page: int
    label: str
    lines: int
    chars: int
    same_page_with_image: bool
    nearby_image_bboxes: list[tuple[float, float, float, float]]


@dataclass
class OrderIssue:
    type: str
    number: int
    first_page: int
    previous_number: int
    previous_first_page: int
    description: str


@dataclass
class PageCoverage:
    page: int
    coverage: float
    text_coverage: float
    image_coverage: float
    text_blocks: int
    image_blocks: int


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def pt_to_in(pt: float) -> float:
    return pt / 72.0


def bbox_area(bbox: tuple[float, float, float, float]) -> float:
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        return 0.0
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])


def intersect_bbox(
    bbox: tuple[float, float, float, float],
    page_rect,
) -> tuple[float, float, float, float]:
    x0 = max(bbox[0], page_rect.x0)
    y0 = max(bbox[1], page_rect.y0)
    x1 = min(bbox[2], page_rect.x1)
    y1 = min(bbox[3], page_rect.y1)
    if x1 <= x0 or y1 <= y0:
        return (0.0, 0.0, 0.0, 0.0)
    return (x0, y0, x1, y1)


def bbox_union_area(bboxes: list[tuple[float, float, float, float]]) -> float:
    """Approximate union area of axis-aligned rectangles via a simple grid.

    This avoids double-counting overlapping text/image bboxes when computing
    page coverage. Resolution: 2 pt cells (sufficient for a layout heuristic).
    """
    if not bboxes:
        return 0.0

    min_x = min(b[0] for b in bboxes)
    min_y = min(b[1] for b in bboxes)
    max_x = max(b[2] for b in bboxes)
    max_y = max(b[3] for b in bboxes)
    grid_res = 2.0
    cols = max(1, int((max_x - min_x) / grid_res) + 1)
    rows = max(1, int((max_y - min_y) / grid_res) + 1)
    cells = set()
    for bbox in bboxes:
        x0, y0, x1, y1 = bbox
        c0 = max(0, int((x0 - min_x) / grid_res))
        c1 = min(cols - 1, int((x1 - min_x) / grid_res))
        r0 = max(0, int((y0 - min_y) / grid_res))
        r1 = min(rows - 1, int((y1 - min_y) / grid_res))
        for c in range(c0, c1 + 1):
            for r in range(r0, r1 + 1):
                cells.add((c, r))
    return len(cells) * grid_res * grid_res


def merge_bboxes(
    bboxes: list[tuple[float, float, float, float]],
) -> list[tuple[float, float, float, float]]:
    """Merge overlapping or touching bboxes."""
    if not bboxes:
        return []

    bboxes = [tuple(float(v) for v in b) for b in bboxes]  # type: ignore[assignment]
    changed = True
    while changed:
        changed = False
        merged: list[tuple[float, float, float, float]] = []
        used = set()
        for i, b1 in enumerate(bboxes):
            if i in used:
                continue
            x0, y0, x1, y1 = b1
            for j in range(i + 1, len(bboxes)):
                if j in used:
                    continue
                c0, d0, c1, d1 = bboxes[j]
                if x0 <= c1 and c0 <= x1 and y0 <= d1 and d0 <= y1:
                    x0 = min(x0, c0)
                    y0 = min(y0, d0)
                    x1 = max(x1, c1)
                    y1 = max(y1, d1)
                    used.add(j)
                    changed = True
            used.add(i)
            merged.append((x0, y0, x1, y1))
        bboxes = merged
    return bboxes


# ---------------------------------------------------------------------------
# Drawing extraction (vector figures are figures in this preprint)
# ---------------------------------------------------------------------------
def get_page_drawing_bboxes(
    page: fitz.Page,
) -> list[tuple[float, float, float, float]]:
    """Return merged bboxes of significant vector drawings on a page."""
    page_rect = page.rect
    bboxes: list[tuple[float, float, float, float]] = []
    try:
        for d in page.get_drawings():
            drect = d.get("rect")
            if drect is None:
                continue
            bbox = intersect_bbox((drect.x0, drect.y0, drect.x1, drect.y1), page_rect)
            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                continue
            if bbox_area(bbox) >= DRAWING_AREA_THRESHOLD_PT2:
                bboxes.append(bbox)
    except Exception:
        pass
    return merge_bboxes(bboxes)


def get_page_image_bboxes(
    page: fitz.Page,
) -> list[tuple[float, float, float, float]]:
    """Return merged bboxes of embedded raster images on a page."""
    page_rect = page.rect
    bboxes: list[tuple[float, float, float, float]] = []
    try:
        for img_info in page.get_images(full=True):
            xref = img_info[0]
            try:
                rects = page.get_image_rects(xref)
            except Exception:
                rects = []
            for rect in rects:
                if rect.is_empty:
                    continue
                bbox = intersect_bbox((rect.x0, rect.y0, rect.x1, rect.y1), page_rect)
                if bbox_area(bbox) > 0:
                    bboxes.append(bbox)
    except Exception:
        pass
    return merge_bboxes(bboxes)


def get_page_visual_bboxes(
    page: fitz.Page,
) -> list[tuple[float, float, float, float]]:
    """Combine raster images and vector drawings into merged visual bboxes."""
    drawing_bboxes = get_page_drawing_bboxes(page)
    image_bboxes = get_page_image_bboxes(page)
    return merge_bboxes(drawing_bboxes + image_bboxes)


# ---------------------------------------------------------------------------
# Core audit functions
# ---------------------------------------------------------------------------
def extract_images(doc: fitz.Document) -> list[ImageRecord]:
    records: list[ImageRecord] = []
    for page_no in range(len(doc)):
        page = doc[page_no]
        page_rect = page.rect
        page_area = page_rect.width * page_rect.height
        page_width_in = pt_to_in(page_rect.width)

        merged = get_page_visual_bboxes(page)

        for bbox in merged:
            width_in = pt_to_in(bbox[2] - bbox[0])
            height_in = pt_to_in(bbox[3] - bbox[1])
            area = bbox_area(bbox)
            area_ratio = area / page_area if page_area else 0.0
            width_ratio = width_in / page_width_in if page_width_in else 0.0

            issue_parts = []
            if width_in < MIN_IMG_WIDTH_IN:
                issue_parts.append("too small")
            if area_ratio > MAX_PAGE_AREA_COVERAGE:
                issue_parts.append("excessive page coverage")

            records.append(
                ImageRecord(
                    page=page_no + 1,
                    bbox=bbox,
                    width_in=round(width_in, 2),
                    height_in=round(height_in, 2),
                    area_ratio=round(area_ratio, 3),
                    width_ratio=round(width_ratio, 3),
                    issue="; ".join(issue_parts) if issue_parts else "OK",
                )
            )
    return records


def extract_captions(
    doc: fitz.Document,
) -> tuple[list[CaptionRecord], dict[str, set[int]]]:
    """Extract caption blocks and return valid caption numbers per category."""
    caption_records: list[CaptionRecord] = []
    # valid numbers for Figure/Table references (filter spurious references)
    valid_numbers: dict[str, set] = {"Figure": set(), "Table": set()}

    # Regex for caption block first line, e.g. "Figure 1:" or "Table 1."
    caption_re = re.compile(r"^(Figure|Table)\s+(\d+)\s*[\:\.]\s*(.*)$", re.IGNORECASE)

    for page_no in range(len(doc)):
        page = doc[page_no]
        pnum = page_no + 1
        visual_bboxes = get_page_visual_bboxes(page)

        for block in page.get_text("blocks"):
            text = block[4]
            lines = text.strip().splitlines()
            if not lines:
                continue
            first_line = lines[0].strip()
            match = caption_re.match(first_line)
            if not match:
                continue

            kind = match.group(1).capitalize()
            num = int(match.group(2))
            # Count remaining lines of the caption block
            n_lines = len(lines)
            n_chars = len(text)
            label = f"{kind} {num}"

            valid_numbers[kind].add(num)

            # Nearby visual bboxes (image or drawing) on the same page
            nearby = list(visual_bboxes)

            caption_records.append(
                CaptionRecord(
                    page=pnum,
                    label=label,
                    lines=n_lines,
                    chars=n_chars,
                    same_page_with_image=bool(nearby),
                    nearby_image_bboxes=nearby,
                )
            )

    return caption_records, valid_numbers


def extract_refs(
    doc: fitz.Document,
    valid_numbers: dict[str, set],
) -> dict[str, dict[int, int]]:
    """Return first page on which each valid Figure/Table number is cited."""
    first_ref: dict[str, dict[int, int]] = {"Figure": {}, "Table": {}}
    ref_re = re.compile(r"\b(Figure|Table)\s+(\d+)\b", re.IGNORECASE)

    for page_no in range(len(doc)):
        page = doc[page_no]
        pnum = page_no + 1
        text = page.get_text()
        for kind, num_str in ref_re.findall(text):
            num = int(num_str)
            kind = kind.capitalize()
            if kind not in first_ref:
                continue
            if num not in valid_numbers.get(kind, set()):
                continue
            if num not in first_ref[kind]:
                first_ref[kind][num] = pnum

    return first_ref


def check_order(first_ref: dict[int, int], label: str) -> list[OrderIssue]:
    """Check that first-reference pages are monotonically non-decreasing."""
    issues: list[OrderIssue] = []
    prev_num = None
    prev_page = None
    for num in sorted(first_ref.keys()):
        page = first_ref[num]
        if prev_num is not None and prev_page is not None and page < prev_page:
            issues.append(
                OrderIssue(
                    type=label,
                    number=num,
                    first_page=page,
                    previous_number=prev_num,
                    previous_first_page=prev_page,
                    description=(
                        f"{label} {num} first referenced on p.{page}, "
                        f"before {label} {prev_num} (p.{prev_page})"
                    ),
                )
            )
        prev_num = num
        prev_page = page
    return issues


def compute_page_coverage(doc: fitz.Document) -> list[PageCoverage]:
    coverages: list[PageCoverage] = []
    for page_no in range(len(doc)):
        page = doc[page_no]
        page_area = page.rect.width * page.rect.height

        # Text blocks
        text_bboxes: list[tuple[float, float, float, float]] = []
        for block in page.get_text("blocks"):
            if len(block) >= 5:
                bbox = (
                    float(block[0]),
                    float(block[1]),
                    float(block[2]),
                    float(block[3]),
                )
                if bbox_area(bbox) > 0:
                    text_bboxes.append(bbox)

        # Visual blocks (drawings + raster images)
        img_bboxes = get_page_visual_bboxes(page)

        all_bboxes = text_bboxes + img_bboxes
        covered_area = bbox_union_area(all_bboxes)
        text_area = bbox_union_area(text_bboxes)
        image_area = bbox_union_area(img_bboxes)
        coverage = covered_area / page_area if page_area else 0.0
        text_cov = text_area / page_area if page_area else 0.0
        image_cov = image_area / page_area if page_area else 0.0

        coverages.append(
            PageCoverage(
                page=page_no + 1,
                coverage=round(coverage, 3),
                text_coverage=round(text_cov, 3),
                image_coverage=round(image_cov, 3),
                text_blocks=len(text_bboxes),
                image_blocks=len(img_bboxes),
            )
        )
    return coverages


def check_supplemental_index(doc: fitz.Document) -> dict:
    """Check whether the document contains a Supplemental figures index."""
    # Scan the whole document for the supplemental index section.
    pages_text: list[tuple[int, str]] = []
    for page_no in range(len(doc)):
        pages_text.append((page_no + 1, doc[page_no].get_text()))
    combined = "\n".join(t for _, t in pages_text)

    has_supplemental_header = bool(
        re.search(r"SUPPLEMENTAL\s+(?:INFORMATION|FIGURES|DATA)", combined, re.IGNORECASE)
    ) or bool(re.search(r"Supplemental\s+(?:Information|Figures|Data)", combined, re.IGNORECASE))

    # Parse explicit ranges like "Figures S1–S12" or "Figures S1-S12"
    range_match = re.search(r"Figures?\s+S(\d+)\s*[\u2013-]\s*S(\d+)", combined, re.IGNORECASE)
    s_figures: dict[str, bool] = {}
    if range_match:
        start_s, end_s = int(range_match.group(1)), int(range_match.group(2))
        for i in range(start_s, end_s + 1):
            s_figures[f"S{i}"] = True

    # Also check individual Figure S1..S12 occurrences
    for i in range(1, 13):
        key = f"S{i}"
        if key not in s_figures:
            s_figures[key] = bool(re.search(rf"Figure\s+S{i}\b", combined, re.IGNORECASE))

    # Legacy S1-S11 coverage (requested by the task)
    s1_s11 = {f"S{i}": s_figures.get(f"S{i}", False) for i in range(1, 12)}

    return {
        "checked_pages": [p for p, _ in pages_text],
        "has_supplemental_header": has_supplemental_header,
        "s_figures_present": s_figures,
        "all_s1_s11_present": all(s1_s11.values()),
        "all_s1_s12_present": all(s_figures.values()),
        "missing_s1_s11": [k for k, v in s1_s11.items() if not v],
        "missing_s1_s12": [k for k, v in s_figures.items() if not v],
        "range_text": range_match.group(0) if range_match else None,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    if not PDF_PATH.exists():
        print(f"PDF not found: {PDF_PATH}", file=sys.stderr)
        return 1

    doc = fitz.open(str(PDF_PATH))
    if len(doc) != 27:
        print(f"Warning: expected 27 pages, got {len(doc)}", file=sys.stderr)

    images = extract_images(doc)
    captions, valid_numbers = extract_captions(doc)
    first_ref = extract_refs(doc, valid_numbers)
    order_issues = check_order(first_ref["Figure"], "Figure") + check_order(
        first_ref["Table"], "Table"
    )
    coverages = compute_page_coverage(doc)
    supplemental = check_supplemental_index(doc)

    # Build report
    report = {
        "pdf_path": str(PDF_PATH),
        "pages": len(doc),
        "page_width_in": round(pt_to_in(doc[0].rect.width), 3),
        "page_height_in": round(pt_to_in(doc[0].rect.height), 3),
        "images": [asdict(r) for r in images],
        "image_issues": [asdict(r) for r in images if r.issue != "OK"],
        "captions": [asdict(c) for c in captions],
        "caption_on_image_page": [asdict(c) for c in captions if c.same_page_with_image],
        "caption_not_on_image_page": [asdict(c) for c in captions if not c.same_page_with_image],
        "first_reference_pages": first_ref,
        "order_issues": [asdict(o) for o in order_issues],
        "page_coverage": [asdict(c) for c in coverages],
        "low_coverage_pages": [c.page for c in coverages if c.coverage < MIN_PAGE_COVERAGE],
        "supplemental_index": supplemental,
    }

    # Write JSON
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Human-readable summary on stdout
    # ------------------------------------------------------------------
    print(f"PDF layout audit: {PDF_PATH}")
    print(
        f'Pages: {len(doc)} | Page size: {report["page_width_in"]}" x {report["page_height_in"]}"'
    )
    print()

    print("=" * 70)
    print("1. IMAGE RENDERING (page, width[in], height[in], issue)")
    print("=" * 70)
    if images:
        for img in images:
            print(
                f'  p.{img.page:>2d}  W={img.width_in:.2f}" H={img.height_in:.2f}" '
                f"area_ratio={img.area_ratio:.2f}  -> {img.issue}"
            )
    else:
        print("  No embedded visual objects detected.")

    print()
    print("=" * 70)
    print("2. CAPTION PROXIMITY (caption label, page, lines, same-page image?)")
    print("=" * 70)
    if captions:
        for c in captions:
            same = "yes" if c.same_page_with_image else "NO"
            print(f"  {c.label:<12} p.{c.page:>2d}  {c.lines} lines  same-page={same}")
    else:
        print("  No captions detected.")

    print()
    print("=" * 70)
    print("3. CITATION ORDER (first reference page for each Figure/Table)")
    print("=" * 70)
    for kind in ("Figure", "Table"):
        refs = first_ref.get(kind, {})
        print(f"  {kind}:")
        if refs:
            for num in sorted(refs.keys()):
                print(f"    {kind} {num}: p.{refs[num]}")
        else:
            print("    none")
    if order_issues:
        print("  Order issues:")
        for o in order_issues:
            print(f"    {o.description}")
    else:
        print("  Order issues: none")

    print()
    print("=" * 70)
    print("4. PAGE COVERAGE (text + image; pages < 30% flagged)")
    print("=" * 70)
    for c in coverages:
        flag = "<-- LOW" if c.coverage < MIN_PAGE_COVERAGE else ""
        print(
            f"  p.{c.page:>2d}  total={c.coverage:.2%} "
            f"(text={c.text_coverage:.2%}, image={c.image_coverage:.2%}) {flag}"
        )

    print()
    print("=" * 70)
    print("5. SUPPLEMENTAL INDEX (last pages)")
    print("=" * 70)
    print(f"  Has Supplemental header: {supplemental['has_supplemental_header']}")
    print(
        "  S1-S12 present: "
        f"{supplemental['all_s1_s12_present']}  "
        f"missing={supplemental['missing_s1_s12']}"
    )
    print(
        "  S1-S11 present: "
        f"{supplemental['all_s1_s11_present']}  "
        f"missing={supplemental['missing_s1_s11']}"
    )
    print(f"  Range text: {supplemental['range_text']}")

    print()
    print("=" * 70)
    print("6. RECOMMENDED FIXES")
    print("=" * 70)
    recommendations: list[str] = []

    # Image sizing
    small = [img for img in images if img.width_in < MIN_IMG_WIDTH_IN]
    large = [img for img in images if "excessive page coverage" in img.issue]
    if small:
        recommendations.append(
            f'{len(small)} image(s) rendered narrower than {MIN_IMG_WIDTH_IN}". '
            'Consider enlarging or combining panels (target at least 3" width).'
        )
    if large:
        recommendations.append(
            f"{len(large)} image(s) cover >80% of the page. "
            "Verify Cell Press single/double-column width rules; resize if necessary."
        )

    # Width overflow beyond the text column (7.0")
    overflow = [img for img in images if img.width_in > PAGE_TEXT_WIDTH_IN + 0.05]
    if overflow:
        labels = ", ".join(f"p.{img.page}" for img in overflow)
        recommendations.append(
            f'Figure(s) wider than the {PAGE_TEXT_WIDTH_IN}" text column: {labels}. '
            "Resize to fit the column width or use a full-width float."
        )

    # Caption proximity
    off_page_captions = [c for c in captions if not c.same_page_with_image]
    if off_page_captions:
        labels = ", ".join(c.label for c in off_page_captions)
        recommendations.append(
            f"Captions on a different page from their figure/table: {labels}. "
            "Move captions to the same page as the visual or use \\clearpage to keep them together."
        )

    # Order issues
    if order_issues:
        recommendations.append(
            "Citation order is not monotonic. Revise manuscript so that "
            "Figure/Table N is first cited before Figure/Table N+1."
        )

    # Low coverage pages
    low_cov_pages = [c.page for c in coverages if c.coverage < MIN_PAGE_COVERAGE]
    if low_cov_pages:
        recommendations.append(
            f"Pages with <30% coverage: {low_cov_pages}. "
            "Check whether these are intentional placeholder/index pages "
            "(e.g. 'MAIN FIGURE TITLES AND LEGENDS', 'MAIN TABLES', supplemental index) "
            "or float-induced blank space; if the latter, add \\clearpage or reposition floats."
        )

    # Supplemental
    if not supplemental["all_s1_s12_present"]:
        recommendations.append(
            f"Supplemental figure index is incomplete; missing {supplemental['missing_s1_s12']}. "
            "Ensure the index lists Figure S1-S12 explicitly."
        )

    if not recommendations:
        recommendations.append("No major layout issues detected.")
    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. {rec}")

    print()
    print(f"Full JSON report written to: {OUT_JSON}")

    doc.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
