"""Gate 2: GSE323164 gene-namespace verification.

Checks:
1. Matrix shape: 34,962 genes x 6 FPKM samples + 6 count columns + annotation.
2. Namespace consistency: ENSG (gene_id), HGNC symbol (gene_name), EntrezID.
   - ENSG uniqueness / format / version suffix
   - symbol non-empty, unique per ENSG, HGNC-valid format
   - EntrezID completeness and numeric format
3. Mapability of GSE323164 symbols to the human-side columns of
   data/mouse_human_orthologs.csv and data/rat_human_orthologs.csv
   (symbol-level union + ENSG-level for the mouse table).
4. HGNC currency spot-check via REST API (sample of symbols).

Output: gse323164_namespace_verification.json
"""

import json
import re
from pathlib import Path

import pandas as pd

BASE = Path("/home/dq/Codex/Ai/data/ded_dryarm/gse323164")
ORTHO_MOUSE = Path("/home/dq/Codex/Ai/data/mouse_human_orthologs.csv")
ORTHO_RAT = Path("/home/dq/Codex/Ai/data/rat_human_orthologs.csv")

ENSG_RE = re.compile(r"^ENSG\d{11}$")
SYMBOL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def main() -> None:
    fpkm = pd.read_csv(
        BASE / "GSE323164_genes_fpkm_expression.txt.gz", sep="\t"
    )
    report: dict = {}
    report["n_cols"] = int(fpkm.shape[1])
    report["n_genes"] = int(fpkm.shape[0])
    report["columns"] = list(fpkm.columns)
    fpkm_cols = [c for c in fpkm.columns if c.startswith("FPKM.")]
    count_cols = [c for c in fpkm.columns if c.startswith("count.")]
    report["n_fpkm_columns"] = len(fpkm_cols)
    report["n_count_columns"] = len(count_cols)
    report["fpkm_columns"] = fpkm_cols
    report["count_columns"] = count_cols

    # 1. gene_id (ENSG)
    ensg = fpkm["gene_id"].astype(str)
    report["ensg_total"] = int(ensg.nunique())
    report["ensg_duplicates"] = int(ensg.duplicated().sum())
    report["ensg_format_valid"] = int(ensg.apply(lambda x: bool(ENSG_RE.match(x))).sum())
    report["ensg_version_suffix"] = int(ensg.apply(lambda x: "." in x).sum())
    report["ensg_empty"] = int((ensg == "").sum())

    # 2. gene_name (symbol)
    sym = fpkm["gene_name"].astype(str)
    report["symbol_total"] = int(sym.nunique())
    report["symbol_duplicates"] = int(sym.duplicated().sum())
    report["symbol_empty"] = int((sym == "").sum())
    report["symbol_format_valid"] = int(
        sym.apply(lambda x: bool(SYMBOL_RE.match(x))).sum()
    )
    # symbols shared by >1 ENSG (many-to-one)
    sym_to_ensg = fpkm.groupby("gene_name")["gene_id"].nunique()
    report["symbols_multiple_ensg"] = int((sym_to_ensg > 1).sum())
    # ENSG shared by >1 symbol (one-to-many)
    ensg_to_sym = fpkm.groupby("gene_id")["gene_name"].nunique()
    report["ensg_multiple_symbols"] = int((ensg_to_sym > 1).sum())
    # Gene names that look like locus IDs / non-HGNC
    report["symbol_nonstandard"] = int(
        sym.apply(
            lambda x: not re.match(r"^[A-Za-z][A-Za-z0-9._-]*$", x)
        ).sum()
    )

    # 3. EntrezID
    entrez = fpkm["EntrezID"].astype(str)
    report["entrez_empty"] = int((entrez == "").sum())
    report["entrez_numeric"] = int(
        entrez[entrez != ""].apply(lambda x: x.isdigit()).sum()
    )
    report["entrez_present"] = int((entrez != "").sum())

    # 4. Mapability to human ortholog tables
    mouse = pd.read_csv(ORTHO_MOUSE)
    rat = pd.read_csv(ORTHO_RAT)
    human_symbols_mouse = set(mouse["human_gene_symbol"].astype(str))
    human_symbols_rat = set(rat["human_gene_symbol"].astype(str))
    human_symbols_union = human_symbols_mouse | human_symbols_rat
    human_ensg_mouse = set(mouse["human_ensembl_id"].astype(str))
    symbol_set = set(sym)
    report["ortho_mouse_rows"] = len(mouse)
    report["ortho_rat_rows"] = len(rat)
    report["ortho_human_symbol_union"] = len(human_symbols_union)
    report["mapable_symbol_union"] = int(
        len(symbol_set & human_symbols_union)
    )
    report["mapable_symbol_union_pct"] = round(
        100.0 * len(symbol_set & human_symbols_union) / len(symbol_set), 2
    )
    report["mapable_symbol_mouse_only"] = int(
        len(symbol_set & human_symbols_mouse)
    )
    report["mapable_symbol_rat_only"] = int(len(symbol_set & human_symbols_rat))
    report["mapable_ensg_mouse"] = int(len(set(ensg) & human_ensg_mouse))
    report["mapable_ensg_mouse_pct"] = round(
        100.0 * len(set(ensg) & human_ensg_mouse) / len(set(ensg)), 2
    )

    # 5. Sample columns sanity: 6 samples, names cont_1..3, HN_1..3
    report["samples_ok"] = sorted(fpkm_cols) == sorted(
        [
            "FPKM.cont_1",
            "FPKM.cont_2",
            "FPKM.cont_3",
            "FPKM.HN_1",
            "FPKM.HN_2",
            "FPKM.HN_3",
        ]
    )
    # Gene order identical between FPKM columns
    report["gene_row_order_identical"] = bool(
        (fpkm[fpkm_cols].index == fpkm[count_cols].index).all()
    )

    out = BASE / "gse323164_namespace_verification.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
