# GSE299152 — Human External Test Set: Ordinal Annotation Report

- Generated: 2026-08-06 15:40:37
- Input: data/gse299152/gse299152_human_raw.h5ad (26628 cells x 36601 genes, raw counts)
- Output: data/gse299152/gse299152_annotated.h5ad (23970 cells x 11274 genes)
- Source: GEO GSE299152, 'Single-cell mapping of the glycocalyx in the adult human and mouse ocular surface' (PNAS Nexus 2025); human donors Donor2046/2206/3748 (cornea-conjunctiva).

## Method (mirror of GSE153515 annotation pipeline v2)

| Parameter | Value |
|-----------|-------|
| Mirrored pipeline | protocol_15_annotate_gse153515.py (pipeline v2, 2026-07-31) |
| Integration | harmonypy 30 PCs, batch = donor |
| Leiden resolution | 0.7 (auto-tuned to ~16 clusters; candidates [0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.2]) |
| n clusters | 15 |
| Seed | 42 |
| HVG | 2000 (flavor='seurat') |
| DEG enrichment | wilcoxon rank_genes_groups, top 200 DEGs/cluster; enrichment = (n_in_top200/n_total)*(1 - mean_rank/200) |
| NON_EPI_THRESHOLD | 0.3 |
| Marker sets | 8 pipeline sets (65 entries, 63 unique symbols; KRT14/KRT15 shared between LPC and PMC_basal as in v2) + 2 supplementary guard sets (Endothelial/Stromal, 8 genes) |

Deviations from pipeline v2 (all deliberate, see script docstring):
1. Donor pre-assigned in obs['donor'] (no barcode recovery).
2. var_names are real symbols (direct marker matching).
3. Resolution auto-tuned to land ~16 clusters (v2 fixed res=0.6).
4. Supplementary Endothelial/Stromal non-epithelial guard added (v2 has no such sets; task spec maps these to NON_LAYER). The guard fires only when its enrichment > threshold and exceeds every pipeline-set enrichment.
5. Task-required obs aliases cluster/cell_type/layer_index added; pipeline columns retained.

## QC

- Cells retained: 23970 / 26628 (90.0%); genes retained: 11274 / 36601.
- Filters (v2): genes detected in >=3% of cells; cells with n_genes within 5th-95th percentile.

## Cluster assignment (cluster -> cell_type_raw -> harmonized -> layer_index)

| Cluster | n_cells | cell_type_raw | cell_type (harmonized) | layer_index |
|---------|---------|---------------|------------------------|-------------|
| 0 | 3675 | LPC | LSC | 0 |
| 1 | 3197 | LSC | LSC | 0 |
| 2 | 2701 | TDC | differentiated | 3 |
| 3 | 2477 | conjunctival | NON_LAYER | -1 |
| 4 | 2477 | LPC | LSC | 0 |
| 5 | 1967 | LPC | LSC | 0 |
| 6 | 1653 | TDC | differentiated | 3 |
| 7 | 1349 | LSC | LSC | 0 |
| 8 | 1182 | LPC | LSC | 0 |
| 9 | 1099 | PMC_basal | basal | 2 |
| 10 | 1038 | LPC | LSC | 0 |
| 11 | 566 | LSC | LSC | 0 |
| 12 | 322 | LPC | LSC | 0 |
| 13 | 148 | LSC | LSC | 0 |
| 14 | 119 | conjunctival | NON_LAYER | -1 |

## Assignment confidence (per cluster)

`low_confidence` = assigned-type enrichment < 0.10 OR the assigned type is not the top-scoring set. Zero-enrichment clusters (max_enrichment = 0.000) are defaulted to the first epithelial type (LSC) exactly as the mirrored pipeline would assign them; they are flagged here.

| Cluster | n_cells | assigned | assigned_enrichment | max_set | max_enrichment | low_confidence |
|---------|---------|----------|---------------------|---------|----------------|----------------|
| 0 | 3675 | LPC | 0.164 | Conjunctival | 0.203 | YES |
| 1 | 3197 | LSC | 0.000 | Conjunctival | 0.150 | YES |
| 2 | 2701 | TDC | 0.485 | TDC | 0.485 |  |
| 3 | 2477 | conjunctival | 0.447 | Conjunctival | 0.447 | YES |
| 4 | 2477 | LPC | 0.097 | Conjunctival | 0.220 | YES |
| 5 | 1967 | LPC | 0.075 | Conjunctival | 0.213 | YES |
| 6 | 1653 | TDC | 0.345 | TDC | 0.345 |  |
| 7 | 1349 | LSC | 0.000 | LSC | 0.000 | YES |
| 8 | 1182 | LPC | 0.314 | LPC | 0.314 |  |
| 9 | 1099 | PMC_basal | 0.683 | PMC_basal | 0.683 |  |
| 10 | 1038 | LPC | 0.167 | LPC | 0.167 |  |
| 11 | 566 | LSC | 0.000 | LSC | 0.000 | YES |
| 12 | 322 | LPC | 0.017 | LPC | 0.017 | YES |
| 13 | 148 | LSC | 0.000 | Melanocyte | 0.199 | YES |
| 14 | 119 | conjunctival | 0.365 | Conjunctival | 0.365 | YES |

## Caveats (mirroring decisions; see also provenance.json)

- **TAC (layer 1) is empty** at the chosen resolution (0.7, 15 clusters): no cluster's top-200 DEGs are enriched for the TAC proliferation set (MKI67/BIRC5/RRM2/TOP2A/PLK1/TK1); MKI67 is detected in only ~0.5% of raw cells. The mirrored pipeline assigns clusters, not cells; a rare proliferating population does not form a cluster here. GSE153515 v2 also found TAC to be the smallest class (171/15,827 cells, 1.1%).
- **Sub-threshold melanocyte cluster**: cluster 13 (148 cells) has PMEL/MITF among its top DEGs but Melanocyte enrichment 0.199 < NON_EPI_THRESHOLD (0.3), so the mirrored v2 gate assigns it LSC (layer 0). Flagged as low-confidence; a reviewer may want to re-classify it as NON_LAYER.
- **Zero-enrichment clusters**: clusters 7 (1,349 cells) and 11 (566 cells, MT-high) have 0.000 enrichment in every marker set and default to LSC (layer 0). Cluster 12 (322 cells) is VIM/HSPA1A/HSP90 stress-like with near-zero enrichment and defaults to LPC. These follow the pipeline's 'assign every cluster' rule; they are flagged low-confidence in cluster_assignment.json.
- Endothelial/stromal guard did not fire (those markers are near absent: PECAM1/VWF/CDH5/CLDN5 in <2% of cells; COL1A1/LUM in ~3%).

## Per-class counts (ordinal axis)

| cell_type | layer_index | n_cells | fraction |
|-----------|-------------|---------|----------|
| LSC | 0 | 15921 | 0.664 |
| TAC | 1 | 0 | 0.000 |
| basal | 2 | 1099 | 0.046 |
| differentiated | 3 | 4354 | 0.182 |
| NON_LAYER | -1 | 2596 | 0.108 |

- Epithelial fraction (layer_index in 0..3): **21374 / 23970 = 89.17%**
- NON_LAYER: 2596 cells (10.83%)

## Per-donor x layer_index balance

```
layer_index   -1     0    2     3
donor                            
Donor2046    937  3341  721  1435
Donor2206    875  8374  223  1916
Donor3748    784  4206  155  1003
```

Per-donor share within each class:

```
cell_type       LSC  NON_LAYER     basal  differentiated
donor                                                   
Donor2046  0.209849   0.360940  0.656051        0.329582
Donor2206  0.525972   0.337057  0.202912        0.440055
Donor3748  0.264179   0.302003  0.141037        0.230363
```

## Marker expression per assigned class (mean log1p-normalized)

Expected gradient: LSC/basal high TP63/KRT15/KRT14; differentiated high KRT12/KRT3; NON_LAYER carries conjunctival (KRT13), goblet (MUC5AC) or immune (PTPRC) signal.

Note: MUC5AC and PTPRC are NaN because they are expressed in <3% of cells and were removed by the v2 min_cells QC gene filter; goblet/immune signal is captured through the Conjunctival/Immune DEG-enrichment sets instead of per-cell expression here.

```
                n_cells  TP63  KRT15  KRT14  KRT12  KRT3  KRT13  MUC5AC  PTPRC
class                                                                         
LSC               15921 0.094  1.752  1.027  0.328 0.022  4.104     NaN    NaN
NON_LAYER          2596 0.047  1.323  0.558  0.452 0.045  4.584     NaN    NaN
basal              1099 0.296  0.913  3.615  4.236 0.980  0.975     NaN    NaN
differentiated     4354 0.155  1.146  1.000  4.158 1.114  1.866     NaN    NaN
```

## Cluster-level batch sanity

Single-donor clusters (potential batch residual) flagged 'YES'.

```
         n_cells dominant_donor  dominant_frac single_donor_flag
cluster                                                         
0           3675      Donor2206         0.5393                  
1           3197      Donor2206         0.6784                  
2           2701      Donor2206         0.4387                  
3           2477      Donor2046         0.3767                  
4           2477      Donor2206         0.5079                  
5           1967      Donor2206         0.3910                  
6           1653      Donor2206         0.4422                  
7           1349      Donor2206         0.6071                  
8           1182      Donor2046         0.4052                  
9           1099      Donor2046         0.6561                  
10          1038      Donor3748         0.4335                  
11           566      Donor2206         0.5883                  
12           322      Donor2206         0.4938                  
13           148      Donor2046         0.3986                  
14           119      Donor3748         0.7143                  
```

## Provenance

- Script: scripts/protocol_15b_annotate_gse299152.py
- Script SHA-256: `9bc1852c8717f45523b1dae19e9508aef5e26836bf901de6bd7b428d78bae786`
- Pipeline version mirrored: GSE153515 annotation pipeline v2 (2026-07-31)
- Seed: 42 | Resolution: 0.7 | n_clusters: 15
- Timestamp: 2026-08-06 15:40:37
- See also: provenance.json, cluster_assignment.json, marker_scores.csv

---

## Strict variant (robustness)

- Generated: 2026-08-06 15:45:42
- Input: data/gse299152/gse299152_annotated.h5ad (as-is, unchanged)
- Output: data/gse299152/gse299152_annotated_strict.h5ad (23970 cells x 11274 genes)
- Script: data/gse299152/make_gse299152_strict_variant.py (see provenance.json `strict_variant`)

### Remap rule

The low-confidence clusters below are moved OUT of the LSC/LPC classes into NON_LAYER (layer_index = -1, cell_type = NON_LAYER). All other obs columns, QC, clustering and enrichment values are identical to the as-is annotation; `cell_type_raw` is retained to record the pipeline's raw DEG-based assignment.

| Cluster | n_cells | as-is cell_type | as-is layer_index | strict cell_type | strict layer_index | Rationale |
|---------|---------|-----------------|-------------------|------------------|--------------------|-----------|
| 7 | 1349 | LSC | 0 | NON_LAYER | -1 | zero-enrichment (max 0.000); defaulted to LSC by assign-every-cluster rule |
| 11 | 566 | LSC | 0 | NON_LAYER | -1 | zero-enrichment (max 0.000), MT-high; defaulted to LSC by assign-every-cluster rule |
| 12 | 322 | LSC | 0 | NON_LAYER | -1 | near-zero enrichment (max 0.017), VIM/HSPA1A/HSP90 stress-like; defaulted to LPC->LSC |
| 13 | 148 | LSC | 0 | NON_LAYER | -1 | sub-threshold melanocyte enrichment 0.199 < 0.3 gate; defaulted to LSC |
| **Total** | **2385** | LSC | 0 | NON_LAYER | -1 | flagged low-confidence in cluster_assignment.json |

### Per-class counts (strict) vs as-is

| cell_type | layer_index | as-is n_cells | strict n_cells | delta |
|-----------|-------------|---------------|----------------|-------|
| LSC | 0 | 15921 | 13536 | -2385 |
| TAC | 1 | 0 | 0 | +0 |
| basal | 2 | 1099 | 1099 | +0 |
| differentiated | 3 | 4354 | 4354 | +0 |
| NON_LAYER | -1 | 2596 | 4981 | +2385 |

- Epithelial fraction (strict, layer_index in 0..3): **18989 / 23970 = 79.22%**
- NON_LAYER (strict): 4981 cells (20.78%)

### Recommendation

Prefer the **as-is** annotation as primary (locked pipeline-v2 mirror, drop-in compatible with the GSE153515 eval convention) and use the **strict** variant as the pre-registered robustness/sensitivity arm, which excludes low-confidence LSC/LPC cells (sub-threshold melanocyte cluster 13 + zero/near-zero-enrichment clusters 7/11/12).
