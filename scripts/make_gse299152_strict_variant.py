#!/usr/bin/env python3
"""Create the strict robustness variant of the GSE299152 ordinal annotation.

Loads the as-is annotation (data/gse299152/gse299152_annotated.h5ad) and
applies a single, pre-registered remap: the four low-confidence clusters
(7, 11, 12, 13) are moved OUT of the LSC/LPC classes into NON_LAYER
(layer_index = -1, cell_type = NON_LAYER). Everything else — QC, clustering,
DEG enrichment, harmonization, obs schema, layers, obsm — is identical to the
as-is annotation.

Remap rationale (documented in artifacts/gse299152_annotation/annotation_report.md
under "Caveats" and the appended "Strict variant" section):

* cluster 13 (148 cells): PMEL/MITF melanocyte enrichment 0.199 <
  NON_EPI_THRESHOLD (0.3), so the mirrored v2 gate defaulted it to LSC;
  flagged low-confidence.
* clusters 7 (1,349) and 11 (566, MT-high): 0.000 enrichment in every marker
  set; defaulted to LSC by the pipeline's assign-every-cluster rule.
* cluster 12 (322, VIM/HSPA1A/HSP90 stress-like): near-zero enrichment
  (0.017); defaulted to LPC -> LSC.

The strict variant keeps cell_type_raw untouched (it records the pipeline's
raw DEG-based assignment); only the harmonized labels
(cell_type / cell_type_harmonized / layer_index / layer / is_epithelial) are
overridden — exactly the reviewer-driven re-classification flagged in the
as-is report.

Outputs:
* data/gse299152/gse299152_annotated_strict.h5ad
* appends the "Strict variant" section to
  artifacts/gse299152_annotation/annotation_report.md
* adds the "strict_variant" key to
  artifacts/gse299152_annotation/provenance.json

The as-is annotation (gse299152_annotated.h5ad), cluster_assignment.json and
the UMAP figures are NOT modified; the script verifies the as-is file's
SHA-256 before and after.

Usage:
    /home/dq/venv-ds/bin/python3 data/gse299152/make_gse299152_strict_variant.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import anndata as ad
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent  # data/gse299152
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_DIR = SCRIPT_DIR
AS_IS_PATH = DATA_DIR / "gse299152_annotated.h5ad"
STRICT_PATH = DATA_DIR / "gse299152_annotated_strict.h5ad"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "gse299152_annotation"
REPORT_PATH = ARTIFACT_DIR / "annotation_report.md"
PROVENANCE_PATH = ARTIFACT_DIR / "provenance.json"

NON_LAYER = "NON_LAYER"
LAYER_ORDER = ["LSC", "TAC", "basal", "differentiated"]

# Cluster -> metadata for the strict remap (n_cells from cluster_assignment.json).
REMAP: dict[str, dict[str, str | int]] = {
    "7": {
        "n_cells": 1349,
        "from": "LSC",
        "reason": "zero-enrichment (max 0.000); defaulted to LSC by assign-every-cluster rule",
    },
    "11": {
        "n_cells": 566,
        "from": "LSC",
        "reason": "zero-enrichment (max 0.000), MT-high; defaulted to LSC by assign-every-cluster rule",
    },
    "12": {
        "n_cells": 322,
        "from": "LSC",
        "reason": "near-zero enrichment (max 0.017), VIM/HSPA1A/HSP90 stress-like; defaulted to LPC->LSC",
    },
    "13": {
        "n_cells": 148,
        "from": "LSC",
        "reason": "sub-threshold melanocyte enrichment 0.199 < 0.3 gate; defaulted to LSC",
    },
}

OBS_COLUMNS = [
    "donor",
    "donor_id",
    "score_LSC",
    "score_LPC",
    "score_TAC",
    "score_PMC_basal",
    "score_TDC",
    "score_Melanocyte",
    "score_Conjunctival",
    "score_Immune",
    "n_genes",
    "leiden",
    "cell_type_raw",
    "cell_type_harmonized",
    "layer",
    "is_epithelial",
    "cluster",
    "cell_type",
    "layer_index",
]


def log(msg: str) -> None:
    """Print a message with a timestamp."""
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


def script_sha256() -> str:
    """SHA-256 of this script's source (provenance)."""
    h = hashlib.sha256()
    h.update(Path(__file__).resolve().read_bytes())
    return h.hexdigest()


def sha256_file(path: Path) -> str:
    """SHA-256 of an arbitrary file (streamed, memory-safe)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def layer_counts(adata: ad.AnnData) -> pd.Series:
    """Counts per layer_index reindexed to [-1, 0, 1, 2, 3] (all present)."""
    return (
        adata.obs["layer_index"]
        .value_counts()
        .reindex(list(range(-1, len(LAYER_ORDER))))
        .fillna(0)
        .astype(int)
    )


def cnt(s: pd.Series, i: int) -> int:
    """Count for layer index i (guaranteed present after reindex)."""
    return int(s.loc[i])


def apply_remap(adata: ad.AnnData) -> tuple[pd.Series, int]:
    """Override the harmonized labels of the low-confidence clusters.

    Returns (mask, n_remapped) where mask selects the remapped cells.
    cell_type_raw is intentionally kept: it records the raw DEG-based
    assignment of the mirrored pipeline.
    """
    obs = adata.obs
    if not (obs["cluster"].astype(str) == obs["leiden"].astype(str)).all():
        raise ValueError("obs['cluster'] does not mirror obs['leiden']")
    mask = obs["cluster"].astype(str).isin(REMAP)
    n_remap = int(mask.sum())
    n_expected = sum(int(v["n_cells"]) for v in REMAP.values())
    if n_remap != n_expected:
        raise ValueError(f"remapped cells {n_remap} != expected {n_expected}")
    if not (obs.loc[mask, "layer_index"] == 0).all():
        raise ValueError("expected every remapped cluster to be layer 0 (LSC)")
    obs.loc[mask, "cell_type"] = NON_LAYER
    obs.loc[mask, "cell_type_harmonized"] = NON_LAYER
    obs.loc[mask, "layer_index"] = -1
    obs.loc[mask, "layer"] = -1
    obs.loc[mask, "is_epithelial"] = False
    return mask, n_remap


def append_report_section(
    adata: ad.AnnData, n_remap: int, ts: str, as_is_counts: pd.Series
) -> None:
    """Append the 'Strict variant' section to annotation_report.md."""
    strict_counts = layer_counts(adata)
    n = adata.n_obs
    n_epi = sum(cnt(strict_counts, i) for i in range(4))
    lines: list[str] = []
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Strict variant (robustness)")
    lines.append("")
    lines.append(f"- Generated: {ts}")
    lines.append("- Input: data/gse299152/gse299152_annotated.h5ad (as-is, unchanged)")
    lines.append(
        f"- Output: data/gse299152/gse299152_annotated_strict.h5ad "
        f"({n} cells x {adata.n_vars} genes)"
    )
    lines.append(
        "- Script: data/gse299152/make_gse299152_strict_variant.py "
        "(see provenance.json `strict_variant`)"
    )
    lines.append("")
    lines.append("### Remap rule")
    lines.append("")
    lines.append(
        "The low-confidence clusters below are moved OUT of the LSC/LPC classes "
        "into NON_LAYER (layer_index = -1, cell_type = NON_LAYER). All other obs "
        "columns, QC, clustering and enrichment values are identical to the as-is "
        "annotation; `cell_type_raw` is retained to record the pipeline's raw "
        "DEG-based assignment."
    )
    lines.append("")
    lines.append(
        "| Cluster | n_cells | as-is cell_type | as-is layer_index | strict cell_type | strict layer_index | Rationale |"
    )
    lines.append(
        "|---------|---------|-----------------|-------------------|------------------|--------------------|-----------|"
    )
    for cl in sorted(REMAP, key=int):
        v = REMAP[cl]
        sub = adata.obs[adata.obs["cluster"].astype(str) == cl]
        lines.append(f"| {cl} | {len(sub)} | {v['from']} | 0 | {NON_LAYER} | -1 | {v['reason']} |")
    lines.append(
        f"| **Total** | **{n_remap}** | LSC | 0 | {NON_LAYER} | -1 | "
        "flagged low-confidence in cluster_assignment.json |"
    )
    lines.append("")
    lines.append("### Per-class counts (strict) vs as-is")
    lines.append("")
    lines.append("| cell_type | layer_index | as-is n_cells | strict n_cells | delta |")
    lines.append("|-----------|-------------|---------------|----------------|-------|")
    for harm in [*LAYER_ORDER, NON_LAYER]:
        li = LAYER_ORDER.index(harm) if harm in LAYER_ORDER else -1
        as_is = cnt(as_is_counts, li)
        strict = cnt(strict_counts, li)
        lines.append(f"| {harm} | {li} | {as_is} | {strict} | {strict - as_is:+d} |")
    lines.append("")
    lines.append(
        f"- Epithelial fraction (strict, layer_index in 0..3): **{n_epi} / {n} = "
        f"{100 * n_epi / n:.2f}%**"
    )
    lines.append(
        f"- NON_LAYER (strict): {cnt(strict_counts, -1)} cells "
        f"({100 * cnt(strict_counts, -1) / n:.2f}%)"
    )
    lines.append("")
    lines.append("### Recommendation")
    lines.append("")
    lines.append(
        "Prefer the **as-is** annotation as primary (locked pipeline-v2 mirror, "
        "drop-in compatible with the GSE153515 eval convention) and use the "
        "**strict** variant as the pre-registered robustness/sensitivity arm, "
        "which excludes low-confidence LSC/LPC cells (sub-threshold melanocyte "
        "cluster 13 + zero/near-zero-enrichment clusters 7/11/12)."
    )
    lines.append("")

    with open(REPORT_PATH, "a") as fh:
        fh.write("\n".join(lines))
    log(f"  Appended 'Strict variant' section to {REPORT_PATH}")


def update_provenance(adata: ad.AnnData, n_remap: int, ts: str) -> None:
    """Add the 'strict_variant' key to provenance.json."""
    with open(PROVENANCE_PATH) as fh:
        prov = json.load(fh)
    prov["strict_variant"] = {
        "script": "data/gse299152/make_gse299152_strict_variant.py",
        "script_sha256": script_sha256(),
        "timestamp": ts,
        "input": str(AS_IS_PATH),
        "output": str(STRICT_PATH),
        "output_shape": [adata.n_obs, adata.n_vars],
        "n_cells_remapped": n_remap,
        "remap_rule": {
            c: {
                "from": str(v["from"]),
                "to": NON_LAYER,
                "n_cells": int(v["n_cells"]),
                "reason": str(v["reason"]),
            }
            for c, v in REMAP.items()
        },
        "note": (
            "Same schema/QC/annotation as the as-is file; only the harmonized "
            "labels (cell_type / cell_type_harmonized / layer_index / layer / "
            "is_epithelial) of the low-confidence clusters 7/11/12/13 are "
            "overridden to NON_LAYER=-1. cell_type_raw retained. As-is annotation "
            "file not modified."
        ),
    }
    with open(PROVENANCE_PATH, "w") as fh:
        json.dump(prov, fh, indent=2, sort_keys=True)
    log(f"  Updated {PROVENANCE_PATH} with 'strict_variant' key")


def verify(
    adata: ad.AnnData,
    mask: pd.Series,
    as_is_labels: pd.DataFrame,
    as_is_counts: pd.Series,
    sha_before: str,
) -> None:
    """Reload the strict h5ad and assert schema, counts and as-is integrity."""
    strict = ad.read_h5ad(STRICT_PATH)
    obs = strict.obs

    if list(obs.columns) != OBS_COLUMNS:
        raise ValueError(f"obs columns changed: {list(obs.columns)}")

    strict_counts = layer_counts(strict)
    expected = as_is_counts.copy()
    expected[0] -= int(mask.sum())
    expected[-1] += int(mask.sum())
    got = strict_counts.to_dict()
    want = expected.to_dict()
    if got != want:
        raise ValueError(f"strict layer_index counts {got} != expected {want}")

    remapped = obs.loc[mask]
    if not (remapped["cell_type"].astype(str) == NON_LAYER).all():
        raise ValueError("remapped cells not all NON_LAYER in cell_type")
    if not (remapped["cell_type_harmonized"].astype(str) == NON_LAYER).all():
        raise ValueError("remapped cells not all NON_LAYER in cell_type_harmonized")
    if not (remapped["layer_index"] == -1).all():
        raise ValueError("remapped cells not all layer_index -1")
    if remapped["is_epithelial"].any():
        raise ValueError("remapped cells not all is_epithelial False")

    untouched = obs.loc[~mask]
    base = as_is_labels.loc[~mask]
    if not (untouched["cell_type"].astype(str) == base["cell_type"].astype(str)).all():
        raise ValueError("non-remapped cells changed cell_type")
    if not (untouched["layer_index"] == base["layer_index"]).all():
        raise ValueError("non-remapped cells changed layer_index")

    sha_after = sha256_file(AS_IS_PATH)
    if sha_after != sha_before:
        raise ValueError("as-is annotation file changed during strict variant creation")

    log("  VERIFIED: strict h5ad reload OK; obs columns intact; as-is file unchanged")
    log("  strict layer_index counts: " + json.dumps({str(k): int(v) for k, v in got.items()}))
    log(f"  as-is SHA-256: {sha_before}")
    log(f"  strict file SHA-256: {sha256_file(STRICT_PATH)}")


def main() -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if STRICT_PATH.exists():
        log(f"ERROR: output already exists: {STRICT_PATH} (remove it to regenerate)")
        sys.exit(1)

    sha_before = sha256_file(AS_IS_PATH)
    log(f"SHA-256 (as-is, before): {sha_before}")
    log(f"Loading as-is annotation: {AS_IS_PATH}")
    adata = ad.read_h5ad(AS_IS_PATH)
    log(f"Loaded: {adata.n_obs} cells x {adata.n_vars} genes")

    as_is_counts = layer_counts(adata)
    log(
        "As-is layer_index counts: " + json.dumps({str(k): int(v) for k, v in as_is_counts.items()})
    )
    as_is_labels = adata.obs[
        ["cell_type", "cell_type_harmonized", "layer_index", "layer", "is_epithelial"]
    ].copy()

    mask, n_remap = apply_remap(adata)
    log(f"Remapped {n_remap} cells (clusters {sorted(REMAP, key=int)}) to {NON_LAYER}")

    adata.uns["annotation"]["strict_variant"] = {
        "remap_rule": {c: {"from": str(v["from"]), "to": NON_LAYER} for c, v in REMAP.items()},
        "n_cells_remapped": n_remap,
        "note": (
            "low-confidence clusters 7/11/12/13 moved from LSC/LPC to NON_LAYER; "
            "cell_type_raw retained"
        ),
    }

    log(f"Saving strict variant: {STRICT_PATH}")
    adata.write_h5ad(STRICT_PATH)

    verify(adata, mask, as_is_labels, as_is_counts, sha_before)
    append_report_section(adata, n_remap, ts, as_is_counts)
    update_provenance(adata, n_remap, ts)

    sha_after = sha256_file(AS_IS_PATH)
    if sha_after != sha_before:
        raise ValueError("as-is annotation file changed after artifacts update")
    log(f"SHA-256 (as-is, after): {sha_after}")
    log("Done.")


if __name__ == "__main__":
    main()
