#!/usr/bin/env python
"""CellRefine corneal marker gene curation for Corneal-FM A3 ablation.

Based on CellRefine (arxiv:2605.07938) post-pretraining approach:
  - L_prototype: contrastive loss using marker-gene-derived prototypes
  - L_lineage:   lineage-aware separation of closely related cell types
  - L_GMVE:      Gaussian mixture prior for embedding structure

This script:
  1. Defines corneal cell-type marker genes (fine → coarse hierarchy)
  2. Defines the lineage hierarchy for ocular surface cell types
  3. Exports marker-gene prototypes in CellRefine-compatible format
  4. Provides the H(cᵢ) ordering function

References:
  - Collin et al. 2021, Ocul Surf — Human corneal single-cell atlas
  - Ligocki et al. 2021, Sci Rep — Adult corneal cell subtypes
  - CellMarker 2.0 + PanglaoDB + MSigDB C8

Usage:
    python scripts/10_cellrefine_markers.py
    python scripts/10_cellrefine_markers.py --output data/cellrefine_prototypes.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# 1. Corneal Cell-Type Marker Genes (fine → coarse specificity)
# ---------------------------------------------------------------------------
# Each cell type maps to a list of (gene, specificity_level) where:
#   specificity_level: 1 = most specific (fine), 3 = most general (coarse)
# The H(cᵢ) function sorts by specificity_level ascending.

CORNEAL_MARKERS: dict[str, list[tuple[str, int, str]]] = {
    # ── Corneal Epithelium ──
    "corneal_superficial": [
        ("KRT12", 1, "cornea-specific keratin"),
        ("KRT3", 1, "cornea-specific keratin"),
        ("CLU", 2, "corneal epithelial stress"),
        ("ALDH3A1", 2, "corneal crystallin"),
        ("KRT5", 3, "basal epithelial"),
        ("EPCAM", 3, "epithelial adhesion"),
    ],
    "corneal_wing": [
        ("KRT12", 1, "cornea-specific keratin"),
        ("KRT3", 1, "cornea-specific keratin"),
        ("TKT", 2, "corneal transketolase"),
        ("SOD2", 2, "mitochondrial antioxidant"),
        ("KRT5", 3, "basal epithelial"),
        ("EPCAM", 3, "epithelial adhesion"),
    ],
    "corneal_basal": [
        ("PAX6", 1, "corneal epithelial master regulator"),
        ("KRT15", 1, "limbal/basal keratin"),
        ("TP63", 2, "epithelial progenitor"),
        ("ITGB4", 2, "hemidesmosome component"),
        ("KRT5", 3, "basal epithelial"),
        ("KRT14", 3, "basal keratin pair"),
    ],
    "TAC1": [
        ("MKI67", 1, "proliferation marker"),
        ("TOP2A", 1, "DNA replication"),
        ("CCNB1", 2, "G2/M transition"),
        ("PCNA", 2, "DNA replication processivity"),
        ("KRT5", 3, "basal epithelial"),
        ("KRT14", 3, "basal keratin pair"),
    ],
    "TAC2": [
        ("MKI67", 1, "proliferation marker"),
        ("TOP2A", 1, "DNA replication"),
        ("CCNB1", 2, "G2/M transition"),
        ("PCNA", 2, "DNA replication processivity"),
        ("KRT5", 3, "basal epithelial"),
        ("KRT14", 3, "basal keratin pair"),
    ],
    "LSC": [  # Limbal Stem Cells
        ("ABCG2", 1, "stem cell side-population"),
        ("KRT15", 1, "limbal stem cell marker"),
        ("TP63", 2, "epithelial progenitor"),
        ("NGFR", 2, "p75 neurotrophin receptor"),
        ("KRT5", 3, "basal epithelial"),
        ("KRT14", 3, "basal keratin pair"),
    ],
    # ── Corneal Stroma ──
    "keratocyte": [
        ("KERA", 1, "keratocan — cornea-specific proteoglycan"),
        ("LUM", 1, "lumican — corneal stroma"),
        ("DCN", 2, "decorin — stromal proteoglycan"),
        ("ALDH3A1", 2, "corneal crystallin"),
        ("COL1A1", 3, "collagen type I"),
        ("VIM", 3, "mesenchymal intermediate filament"),
    ],
    # ── Corneal Endothelium ──
    "corneal_endothelial": [
        ("SLC4A11", 1, "corneal endothelial transporter"),
        ("COL8A2", 1, "Descemet membrane collagen"),
        ("TJP1", 2, "tight junction protein ZO-1"),
        ("CDH2", 2, "N-cadherin — endothelial adhesion"),
        ("VIM", 3, "mesenchymal intermediate filament"),
        ("CDH1", 3, "E-cadherin — epithelial adhesion"),
    ],
    # ── Conjunctival Epithelium ──
    "conjunctival_epithelial": [
        ("KRT4", 1, "conjunctiva-specific keratin"),
        ("KRT13", 1, "conjunctiva keratin pair"),
        ("MUC5AC", 2, "mucin — shared with goblet"),
        ("KRT19", 2, "glandular/secretory keratin"),
        ("KRT5", 3, "basal epithelial"),
        ("EPCAM", 3, "epithelial adhesion"),
    ],
    "goblet_cell": [
        ("MUC5AC", 1, "goblet-cell mucin"),
        ("SPDEF", 1, "goblet cell master regulator"),
        ("KRT7", 2, "simple epithelial keratin"),
        ("TFF3", 2, "trefoil factor — mucosal protection"),
        ("KRT19", 3, "glandular/secretory keratin"),
        ("EPCAM", 3, "epithelial adhesion"),
    ],
    # ── Immune ──
    "immune_cell": [
        ("CD3D", 1, "T-cell receptor component"),
        ("CD14", 1, "monocyte/macrophage marker"),
        ("MS4A1", 1, "CD20 — B-cell marker"),
        ("NKG7", 1, "NK / cytotoxic T cell"),
        ("PTPRC", 2, "CD45 — pan-leukocyte"),
    ],
    # ── Vascular ──
    "vascular_endothelial": [
        ("PECAM1", 1, "CD31 — endothelial adhesion"),
        ("CDH5", 1, "VE-cadherin — vascular endothelial"),
        ("VWF", 2, "von Willebrand factor"),
        ("CLDN5", 2, "tight junction — blood vessel"),
        ("VIM", 3, "mesenchymal intermediate filament"),
    ],
}

# ---------------------------------------------------------------------------
# 2. Lineage Hierarchy (for L_lineage loss in CellRefine)
# ---------------------------------------------------------------------------
# Format: (cell_type, parent_cell_type) — used to define which cell types
# are "close" and should be separated by L_lineage.

LINEAGE_EDGES: list[tuple[str, str | None]] = [
    # ── Corneal Epithelium differentiation axis (stem→differentiated) ──
    # CellRefine L_lineage: penalizes pairs with the SAME parent in embedding space.
    # TAC1 ↔ TAC2 (both transient amplifying — highly confusable)
    ("TAC1", "LSC"),
    ("TAC2", "LSC"),
    # Basal ↔ Wing ↔ Superficial (differentiation gradient — similar expression)
    ("corneal_superficial", "corneal_wing"),
    ("corneal_wing", "corneal_basal"),
    ("corneal_basal", "LSC"),
    # Root of corneal epithelium
    ("LSC", None),
    # ── Conjunctival lineage ──
    ("goblet_cell", "conjunctival_epithelial"),
    ("conjunctival_epithelial", None),
    # ── Independent lineages ──
    ("keratocyte", None),
    ("corneal_endothelial", None),
    ("vascular_endothelial", None),
    ("immune_cell", None),
]

# ---------------------------------------------------------------------------
# 3. CellRefine H(cᵢ) Ordering Function
# ---------------------------------------------------------------------------


def order_markers_by_specificity(
    markers: list[tuple[str, int, str]],
) -> list[str]:
    """Sort marker genes by specificity (finest → coarsest) per CellRefine H(cᵢ).

    CellRefine requires markers ordered from most cell-type-specific (fine)
    to least specific (coarse), so the prototype encoder captures the
    specificity gradient through positional encoding.

    Parameters
    ----------
    markers : list[tuple[str, int, str]]
        List of (gene_symbol, specificity_level, description) tuples.
        specificity_level: 1 = finest, 3 = coarsest.

    Returns
    -------
    list[str]
        Ordered gene symbols: [finest, ..., coarsest].
    """
    sorted_markers = sorted(markers, key=lambda x: x[1])
    return [gene for gene, _, _ in sorted_markers]


def build_prototype(
    cell_type: str,
    genes: list[str],
) -> dict[str, Any]:
    """Build a CellRefine-compatible prototype entry.

    Parameters
    ----------
    cell_type : str
        Cell type name (must match CELLxGENE author annotation).
    genes : list[str]
        Ordered gene symbols (finest → coarsest).

    Returns
    -------
    dict
        Prototype entry with ``cell_type``, ``genes``, and ``n_genes``.
    """
    return {
        "cell_type": cell_type,
        "genes": genes,
        "n_genes": len(genes),
    }


def build_lineage_tree(
    edges: list[tuple[str, str | None]],
) -> dict[str, str | None]:
    """Build a parent-child lineage map from edge list.

    Parameters
    ----------
    edges : list[tuple[str, str | None]]
        (child, parent) pairs. ``None`` parent = root node.

    Returns
    -------
    dict[str, str | None]
        Mapping: cell_type → parent_cell_type (or None for root).
    """
    tree: dict[str, str | None] = {}
    for child, parent in edges:
        tree[child] = parent
    return tree


# ---------------------------------------------------------------------------
# 4. Export
# ---------------------------------------------------------------------------


def export_prototypes(
    output_path: str | Path,
    markers: dict[str, list[tuple[str, int, str]]] | None = None,
    edges: list[tuple[str, str | None]] | None = None,
) -> None:
    """Export marker-gene prototypes and lineage tree as JSON.

    Parameters
    ----------
    output_path : str | Path
        Destination JSON file path.
    markers : dict | None
        Cell-type → marker gene tuples. Defaults to CORNEAL_MARKERS.
    edges : list | None
        Lineage edge list. Defaults to LINEAGE_EDGES.
    """
    if markers is None:
        markers = CORNEAL_MARKERS
    if edges is None:
        edges = LINEAGE_EDGES

    prototypes = {}
    for ct, mg in markers.items():
        ordered_genes = order_markers_by_specificity(mg)
        prototypes[ct] = build_prototype(ct, ordered_genes)

    lineage_tree = build_lineage_tree(edges)

    output = {
        "description": "CellRefine corneal marker gene prototypes for Corneal-FM A3 ablation",
        "references": [
            "Collin et al. 2021, Ocul Surf — Human corneal single-cell atlas",
            "Ligocki et al. 2021, Sci Rep — Adult corneal cell subtypes",
            "CellRefine: Weerasekara et al. 2026, arXiv:2605.07938",
        ],
        "n_cell_types": len(prototypes),
        "n_lineage_edges": len(edges),
        "domain": "ocular_surface",
        "prototypes": prototypes,
        "lineage_tree": lineage_tree,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2, ensure_ascii=False)

    print(f"✅ Exported {len(prototypes)} prototypes + {len(edges)} lineage edges")
    print(f"   → {output_path}")


# ---------------------------------------------------------------------------
# 5. Print Summary (human-readable)
# ---------------------------------------------------------------------------


def print_summary() -> None:
    """Print a human-readable summary of all marker gene prototypes."""
    print("=" * 72)
    print("  CellRefine Corneal Marker Gene Prototypes")
    print("=" * 72)

    for ct, mg in CORNEAL_MARKERS.items():
        ordered = order_markers_by_specificity(mg)
        fine = [g for g, l, _ in mg if l == 1]
        medium = [g for g, l, _ in mg if l == 2]
        coarse = [g for g, l, _ in mg if l == 3]
        print(f"\n  [{ct}] ({len(mg)} genes)")
        print(f"    Fine:    {', '.join(fine)}")
        print(f"    Medium:  {', '.join(medium)}")
        print(f"    Coarse:  {', '.join(coarse)}")
        print(f"    Ordered: {' → '.join(ordered[:6])}" + (" ..." if len(ordered) > 6 else ""))

    print(f"\n{'─' * 72}")
    print("  Lineage Tree")
    print(f"{'─' * 72}")

    lineage = build_lineage_tree(LINEAGE_EDGES)
    roots = [ct for ct, parent in lineage.items() if parent is None]
    print(f"  Root nodes: {', '.join(roots)}")

    def _print_tree(ct: str, indent: int = 2) -> None:
        prefix = "  " * indent
        children = [c for c, p in LINEAGE_EDGES if p == ct]
        if children:
            print(f"{prefix}├─ {ct}")
            for child in children:
                _print_tree(child, indent + 1)
        else:
            print(f"{prefix}└─ {ct} (leaf)")

    for root in roots:
        _print_tree(root)

    print(f"\n  Total: {len(CORNEAL_MARKERS)} cell types, {len(LINEAGE_EDGES)} lineage edges\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="CellRefine corneal marker gene curation for Corneal-FM A3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/10_cellrefine_markers.py
  python scripts/10_cellrefine_markers.py --output data/cellrefine_prototypes.json
  python scripts/10_cellrefine_markers.py --summary-only
""",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/cellrefine_prototypes.json",
        help="Output JSON path (default: data/cellrefine_prototypes.json)",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Print summary only, do not export JSON.",
    )
    args = parser.parse_args()

    print_summary()

    if not args.summary_only:
        export_prototypes(args.output)
        print("\n📦 Ready for CellRefine post-pretraining (A3 ablation).")
        print("   Load with: cellrefine.load_prototypes('data/cellrefine_prototypes.json')")


if __name__ == "__main__":
    main()
