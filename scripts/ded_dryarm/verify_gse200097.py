"""Verify GSE200097 unpacking and sample/condition attribution (Gate 1).

Cross-checks:
1. GEO series matrix -> 40-sample table (GSM, title, assay, treatment, time).
2. GEO raw_counts header (40 cols) vs local rnaSeq_counts (20 cols): local file
   must equal the RNA-seq subset of the GEO file (gene order and all values).
3. Local gse200097_sample_metadata.tsv vs series matrix rows for the 20 RNA-seq
   samples (GSM id, condition, timepoint agreement).

Outputs:
- gse200097_attribution.tsv  (sample_id, condition, treatment, assay, timepoint)
- gse200097_verification.json (per-step PASS/FAIL + counts)
"""

import gzip
import hashlib
import json
import re
from pathlib import Path

import pandas as pd

BASE = Path("<repo>/data/ded_dryarm/gse200097")


def sha256(path: Path) -> str:
    """SHA-256 of raw file bytes (gzip-compressed bytes for .gz files)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_series_matrix(path: Path) -> pd.DataFrame:
    """Parse GEO series matrix sample tables into a DataFrame."""
    fields: dict[str, list[str]] = {}
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            if not line.startswith("!Sample_"):
                continue
            key, _, rest = line.rstrip("\n").partition("\t")
            if key.startswith("!Sample_series_id"):
                continue
            base_key = key
            n = 0
            while key in fields:
                n += 1
                key = f"{base_key}.{n}"
            values = [v.strip('"') for v in rest.split("\t")]
            fields[key] = values
    df = pd.DataFrame(fields)
    return df


def sha256_plain(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    report: dict = {}

    # 1. Series matrix
    sm = parse_series_matrix(BASE / "GSE200097_series_matrix.txt.gz")
    report["series_matrix_samples"] = len(sm)
    sm["assay"] = sm["!Sample_title"].apply(
        lambda t: "RiboSeq" if "Ribosome profiling" in t else "RNA-seq"
    )
    sm["condition_raw"] = sm["!Sample_characteristics_ch1"].apply(
        lambda v: v.strip() if isinstance(v, str) else ""
    )
    sm["time_raw"] = sm["!Sample_characteristics_ch1.1"].apply(
        lambda v: v.strip() if isinstance(v, str) else ""
    )
    # Normalize treatment
    def norm_treatment(v: str) -> str:
        if "control" in v:
            return "control"
        if "Torin1" in v:
            return "500mOsm+Torin1"
        if "MeAIB" in v:
            return "500mOsm+MeAIB"
        return "500mOsm"

    sm["treatment"] = sm["condition_raw"].apply(norm_treatment)
    sm["timepoint"] = sm["time_raw"].apply(
        lambda v: v.replace("time: ", "") if v else "0h"
    )
    report["rna_seq_samples"] = int((sm["assay"] == "RNA-seq").sum())
    report["riboseq_samples"] = int((sm["assay"] == "RiboSeq").sum())

    # 2. GEO raw_counts header
    with gzip.open(BASE / "GSE200097_raw_counts.txt.gz", "rt") as fh:
        geo_header = fh.readline().rstrip("\n").split("\t")
    report["geo_raw_counts_cols"] = len(geo_header)
    geo_rna_cols = [c for c in geo_header if c.startswith("rnaSeq_")]
    report["geo_rna_cols"] = len(geo_rna_cols)
    report["geo_ribo_cols"] = len([c for c in geo_header if c.startswith("riboSeq_")])

    # 3. Local rnaSeq counts
    with gzip.open(BASE / "GSE200097_rnaSeq_counts.txt.gz", "rt") as fh:
        local_header = fh.readline().rstrip("\n").split("\t")
        local_lines = fh.read().splitlines()
    report["local_rna_counts_cols"] = len(local_header)
    report["local_genes"] = len(local_lines)

    # 4. Exact value comparison: local file == GEO raw_counts RNA-seq subset
    geo_all = pd.read_csv(
        BASE / "GSE200097_raw_counts.txt.gz", sep="\t", index_col=0
    )
    geo_rna = geo_all[[c for c in geo_all.columns if c.startswith("rnaSeq_")]]
    local = pd.read_csv(
        BASE / "GSE200097_rnaSeq_counts.txt.gz", sep="\t", index_col=0
    )
    report["local_columns_match_geo_rna_subset"] = bool(
        list(local.columns) == list(geo_rna.columns)
    )
    report["local_index_matches_geo"] = bool(
        (local.index == geo_rna.index).all()
    )
    report["local_values_equal_geo_rna"] = bool(
        local.equals(geo_rna)
    )

    # 5. Local metadata tsv vs series matrix
    meta = pd.read_csv(BASE / "gse200097_sample_metadata.tsv", sep="\t")
    report["metadata_rows"] = len(meta)
    sm_rna = sm[sm["assay"] == "RNA-seq"].copy()
    sm_rna["sample_name"] = sm_rna["!Sample_title"].apply(
        lambda t: re.sub(r"\s+", "_", t.replace(" ", "_")).lower()
    )
    # Rebuild sample_name from title -> rnaSeq_<cond>_repN pattern
    def sample_name_from_title(t: str) -> str:
        m = re.match(r"Control rep(\d+)", t)
        if m:
            return f"rnaSeq_Control_rep{m.group(1)}"
        m = re.match(r"(\d+)h 500 mOsm(?: \+ (\w+))? rep(\d+)", t)
        if m:
            cond = f"{m.group(1)}hOsm"
            if m.group(2):
                cond = m.group(2)
            return f"rnaSeq_{cond}_rep{m.group(3)}"
        return t

    sm_rna["sample_name"] = sm_rna["!Sample_title"].apply(sample_name_from_title)
    merged = meta.merge(
        sm_rna,
        left_on="sample_name",
        right_on="sample_name",
        how="outer",
        indicator=True,
    )
    report["metadata_merge_both"] = int((merged["_merge"] == "both").sum())
    report["metadata_merge_meta_only"] = int(
        (merged["_merge"] == "left_only").sum()
    )
    report["metadata_merge_sm_only"] = int(
        (merged["_merge"] == "right_only").sum()
    )

    # Condition agreement (local vs GEO treatment/time)
    cond_map = {
        "Control": "control",
        "1hOsm": "500mOsm",
        "6hOsm": "500mOsm",
        "Torin1": "500mOsm+Torin1",
        "MeAIB": "500mOsm+MeAIB",
    }
    time_map = {"0h": "0h", "1h": "1 hour", "6h": "6 hours"}
    # meta columns: condition, timepoint, sample_id
    # sm_rna columns: treatment, timepoint (GEO-normalized), !Sample_geo_accession
    report["condition_agreement"] = int(
        (
            merged["treatment"]
            == merged["condition"].map(cond_map)
        ).sum()
    )
    report["timepoint_agreement"] = int(
        (
            merged["timepoint_y"]
            == merged["timepoint_x"].map(time_map)
        ).sum()
    )
    report["gsm_agreement"] = int(
        (merged["!Sample_geo_accession"] == merged["sample_id"]).sum()
    )

    # 6. Build attribution table
    attr = pd.DataFrame(
        {
            "sample_id": sm_rna["!Sample_geo_accession"],
            "sample_name": sm_rna["sample_name"],
            "assay": "RNA-seq",
            "condition": sm_rna["treatment"].map(
                lambda t: "hyperosmolar" if t != "control" else "control"
            ),
            "treatment": sm_rna["treatment"].map(
                lambda t: {"500mOsm+Torin1": "Torin1", "500mOsm+MeAIB": "MeAIB"}.get(t, "none")
            ),
            "osmolarity": sm_rna["treatment"].map(
                lambda t: "500 mOsm" if t != "control" else "isotonic"
            ),
            "timepoint": sm_rna["timepoint"],
            "is_primary_analysis": sm_rna["sample_name"].map(
                lambda n: "Torin1" not in n and "MeAIB" not in n
            ),
        }
    )
    attr["is_primary_analysis"] = attr["is_primary_analysis"].map(
        lambda b: "TRUE" if b else "FALSE"
    )
    attr_path = BASE / "gse200097_attribution.tsv"
    attr.to_csv(attr_path, sep="\t", index=False)

    # 7. SHA-256 of all gate files (raw file bytes)
    hashes = {}
    for p in sorted(BASE.glob("*")):
        if p.is_file() and p.name not in (
            "gse200097_attribution.tsv",
            "gse200097_verification.json",
        ):
            hashes[p.name] = sha256(p)
    report["sha256"] = hashes

    report_path = BASE / "gse200097_verification.json"
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
