#!/usr/bin/env python3
"""
ensembl_orthologs.py — Download Ensembl mouse → human 1:1 ortholog mapping via BioMart.

Output: data/mouse_human_orthologs.csv with columns:
  mouse_gene_symbol, mouse_ensembl_id, human_ensembl_id, human_gene_symbol

Ensembl Release 113. 1:1 orthologs only.
"""

import csv
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
OUTPUT_CSV = os.path.join(DATA_DIR, "mouse_human_orthologs.csv")

BIOMART_URL = "http://www.ensembl.org/biomart/martservice"

# BioMart XML query: Mus musculus genes with Homo sapiens 1:1 orthologs
BIOMART_XML = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE Query>
<Query virtualSchemaName="default" formatter="TSV" header="0" uniqueRows="0" count="" datasetConfigVersion="0.6">
    <Dataset name="mmusculus_gene_ensembl" interface="default">
        <Filter name="with_hsapiens_homolog" excluded="0"/>
        <Attribute name="external_gene_name"/>
        <Attribute name="ensembl_gene_id"/>
        <Attribute name="hsapiens_homolog_ensembl_gene"/>
        <Attribute name="hsapiens_homolog_associated_gene_name"/>
        <Attribute name="hsapiens_homolog_orthology_type"/>
    </Dataset>
</Query>"""


def fetch_biomart(xml_query: str, max_retries: int = 2) -> str:
    """Fetch TSV from BioMart with retry logic."""
    params = urllib.parse.urlencode({"query": xml_query}).encode("utf-8")
    
    for attempt in range(1, max_retries + 1):
        try:
            req = urllib.request.Request(BIOMART_URL, data=params)
            with urllib.request.urlopen(req, timeout=120) as resp:
                raw = resp.read().decode("utf-8")
            return raw
        except (urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
            print(f"  Attempt {attempt}/{max_retries} failed: {exc}", file=sys.stderr)
            if attempt < max_retries:
                time.sleep(5)

    raise RuntimeError(f"BioMart query failed after {max_retries} attempts")


def parse_tsv(raw: str) -> list[dict[str, str]]:
    """Parse BioMart TSV output and keep only 1:1 orthologs."""
    rows: list[dict[str, str]] = []
    seen_pairs: set[tuple[str, str]] = set()
    
    for line in raw.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        # TSV columns: mouse_symbol, mouse_id, human_id, human_symbol, orthology_type
        parts = line.split("\t")
        if len(parts) < 5:
            continue
        
        mouse_symbol, mouse_id, human_id, human_symbol, orth_type = [p.strip() for p in parts[:5]]
        
        # Keep only 1:1 orthologs
        if orth_type.lower() != "ortholog_one2one":
            continue
        
        # Skip rows with empty human symbol (some have ensembl ID but no gene name)
        if not human_symbol or human_symbol == "":
            continue
        if not mouse_symbol or mouse_symbol == "":
            continue
        
        # Ensure no duplicate mouse→human pairs
        pair = (mouse_id, human_id)
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        
        rows.append({
            "mouse_gene_symbol": mouse_symbol,
            "mouse_ensembl_id": mouse_id,
            "human_ensembl_id": human_id,
            "human_gene_symbol": human_symbol,
        })
    
    return rows


def write_csv(rows: list[dict[str, str]], path: str) -> None:
    """Write rows to CSV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["mouse_gene_symbol", "mouse_ensembl_id", "human_ensembl_id", "human_gene_symbol"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Written: {path} ({len(rows)} rows)")


def main():
    print("=" * 60)
    print("Ensembl Mouse → Human 1:1 Ortholog Downloader")
    print("=" * 60)
    
    # Step 1: Fetch from BioMart
    print("\n[1/3] Querying Ensembl BioMart...")
    print(f"  URL: {BIOMART_URL}")
    try:
        raw_tsv = fetch_biomart(BIOMART_XML)
    except Exception as exc:
        print(f"\n  ERROR: BioMart query failed after retries: {exc}", file=sys.stderr)
        print("  Creating placeholder file...", file=sys.stderr)
        os.makedirs(DATA_DIR, exist_ok=True)
        # Write minimal placeholder
        with open(OUTPUT_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["mouse_gene_symbol", "mouse_ensembl_id", "human_ensembl_id", "human_gene_symbol"])
            writer.writeheader()
        print(f"  Written (empty): {OUTPUT_CSV}")
        sys.exit(1)
    
    total_lines = len([l for l in raw_tsv.strip().splitlines() if l.strip()])
    print(f"  Received {total_lines} raw lines from BioMart")
    
    # Step 2: Parse / filter
    print("\n[2/3] Parsing and filtering 1:1 orthologs...")
    rows = parse_tsv(raw_tsv)
    
    mouse_genes = len(set(r["mouse_ensembl_id"] for r in rows))
    all_raw_mouse = len(set(
        parts[1].strip() for line in raw_tsv.strip().splitlines()
        if line.strip() and (parts := line.split("\t")) and len(parts) >= 2
    ))
    
    pct = (mouse_genes / all_raw_mouse * 100) if all_raw_mouse > 0 else 0
    print(f"  Raw mouse genes with any ortholog data: {all_raw_mouse}")
    print(f"  1:1 human ortholog mappings: {len(rows)}")
    print(f"  Unique mouse genes mapped: {mouse_genes} ({pct:.1f}%)")
    
    # Verify no duplicate mouse→human pairs
    print("\n[3/3] Writing CSV...")
    write_csv(rows, OUTPUT_CSV)
    
    # Summary
    print("\n" + "=" * 60)
    print(f"SUMMARY: Ortholog mapping: {mouse_genes} mouse genes, "
          f"{len(rows)} have 1:1 human orthologs ({pct:.1f}%)")
    print("=" * 60)


if __name__ == "__main__":
    main()
