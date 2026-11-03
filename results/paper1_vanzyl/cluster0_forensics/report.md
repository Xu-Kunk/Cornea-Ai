# Cluster 0 Forensics + TAC Absence — Evidence Report

- Data: `vanzyl_annotated.h5ad` (Stage 1, <compute-node>), 80817 cells x 13250 genes after QC.
- Cluster 0: 13584 cells (16.81% of all cells).
- Statistics: annotation counts + counts-layer expression only (%exp, mean log1p). Read-only.

## A. Cluster 0 source profile

### A.1. partition composition (cluster 0 vs all cells)

| value | c0 n | c0 % | all n | all % | fold vs all |
|---|---|---|---|---|---|
| tm_wedge | 13584 | 100.0 | 45814 | 56.7 | 1.76x |
| central_cornea | 0 | 0.0 | 35003 | 43.3 | 0.00x |

### A.2. donor composition (cluster 0 vs all cells)

| value | c0 n | c0 % | all n | all % | fold vs all |
|---|---|---|---|---|---|
| sanes2022_Hu0216 | 5122 | 37.7 | 26208 | 32.4 | 1.16x |
| sanes2022_Hu0220 | 4418 | 32.5 | 23806 | 29.5 | 1.10x |
| sanes2022_Hu0218 | 2503 | 18.4 | 15521 | 19.2 | 0.96x |
| sanes2022_Hu0235 | 1541 | 11.3 | 9847 | 12.2 | 0.93x |
| sanes2022_Pt14 | 0 | 0.0 | 5435 | 6.7 | 0.00x |

### A.3. sample composition (cluster 0, top 10)

| sample | c0 n | c0 % | all n | all % | fold vs all |
|---|---|---|---|---|---|
| 8530a13f-00f4-48be-80d1-e0cf7e20f326 | 3014 | 22.2 | 5932 | 7.3 | 3.02x |
| 00573bbd-d15f-4e1b-acf5-0876f760ad8a | 2631 | 19.4 | 6102 | 7.6 | 2.57x |
| 1c89a575-306c-4fbc-a8c4-4642a828f052 | 2473 | 18.2 | 5581 | 6.9 | 2.64x |
| 99f1a14b-b792-4090-b41c-416a1ae76aa7 | 2277 | 16.8 | 4957 | 6.1 | 2.73x |
| d87ca80d-f5a7-4869-b21d-cbaec5039fcb | 1536 | 11.3 | 3142 | 3.9 | 2.91x |
| e7efd831-877f-492a-af26-ecd53bd99fba | 725 | 5.3 | 5021 | 6.2 | 0.86x |
| 82d3c950-1072-4331-887c-2de485c949b1 | 679 | 5.0 | 4762 | 5.9 | 0.85x |
| 18bbf879-53e3-40cd-88b0-f1411ae7dc10 | 226 | 1.7 | 4920 | 6.1 | 0.27x |
| 80b5b483-98a5-4fd2-8fd5-24f2cbb2892a | 18 | 0.1 | 3421 | 4.2 | 0.03x |
| 8312d914-2f3a-4bcc-8c7e-f7ce6937e314 | 5 | 0.0 | 1976 | 2.4 | 0.02x |

cluster 0 spans 10 samples (all data spans 16 samples).

### A.4. author_cell_type composition (cluster 0 vs all cells)

| value | c0 n | c0 % | all n | all % | fold vs all |
|---|---|---|---|---|---|
| K_Epi-Wing | 0 | 0.0 | 15799 | 19.5 | 0.00x |
| Ciliary_Muscle | 13477 | 99.2 | 13592 | 16.8 | 5.90x |
| K_Epi-Basal | 0 | 0.0 | 8860 | 11.0 | 0.00x |
| K_Epi-Superficial | 0 | 0.0 | 6968 | 8.6 | 0.00x |
| K_Endo | 0 | 0.0 | 5886 | 7.3 | 0.00x |
| Uveal_Fibro | 2 | 0.0 | 5746 | 7.1 | 0.00x |
| Schwann | 1 | 0.0 | 3815 | 4.7 | 0.00x |
| TM_Fibro | 76 | 0.6 | 2809 | 3.5 | 0.16x |
| CC_VenEndo | 0 | 0.0 | 1733 | 2.1 | 0.00x |
| Conj_Melanocyte | 0 | 0.0 | 1627 | 2.0 | 0.00x |
| Macrophage | 2 | 0.0 | 1444 | 1.8 | 0.01x |
| K_Fibro | 0 | 0.0 | 1269 | 1.6 | 0.00x |
| Uveal_Melanocyte | 0 | 0.0 | 1222 | 1.5 | 0.00x |
| Limbal_Epi-Basal | 0 | 0.0 | 1117 | 1.4 | 0.00x |
| Limbal_Epi-Wing | 0 | 0.0 | 976 | 1.2 | 0.00x |
| Conj_Epi-Basal | 0 | 0.0 | 927 | 1.1 | 0.00x |
| Conj_Epi-Superficial | 0 | 0.0 | 895 | 1.1 | 0.00x |
| Conj_Epi-Wing | 0 | 0.0 | 864 | 1.1 | 0.00x |
| Lymphatic_Endo | 0 | 0.0 | 824 | 1.0 | 0.00x |
| Limbal_Epi-Superficial | 0 | 0.0 | 716 | 0.9 | 0.00x |
| VascEndo | 0 | 0.0 | 714 | 0.9 | 0.00x |
| Sclera_Fibro | 0 | 0.0 | 657 | 0.8 | 0.00x |
| Schlemm_Endo | 0 | 0.0 | 609 | 0.8 | 0.00x |
| Lymphocyte | 13 | 0.1 | 468 | 0.6 | 0.17x |
| Pericyte | 13 | 0.1 | 402 | 0.5 | 0.19x |
| K_Epi-TA | 0 | 0.0 | 332 | 0.4 | 0.00x |
| Goblet | 0 | 0.0 | 279 | 0.3 | 0.00x |
| FibroX | 0 | 0.0 | 164 | 0.2 | 0.00x |
| Immune | 0 | 0.0 | 82 | 0.1 | 0.00x |
| Mast | 0 | 0.0 | 21 | 0.0 | 0.00x |

### A.4b. cell_type composition (cluster 0 vs all cells)

| value | c0 n | c0 % | all n | all % | fold vs all |
|---|---|---|---|---|---|
| corneal epithelial cell | 0 | 0.0 | 22767 | 28.2 | 0.00x |
| ciliary muscle cell | 13477 | 99.2 | 13592 | 16.8 | 5.90x |
| basal cell | 0 | 0.0 | 11669 | 14.4 | 0.00x |
| fibroblast | 78 | 0.6 | 10645 | 13.2 | 0.04x |
| corneal endothelial cell | 0 | 0.0 | 5886 | 7.3 | 0.00x |
| Schwann cell | 1 | 0.0 | 3815 | 4.7 | 0.00x |
| melanocyte | 0 | 0.0 | 2849 | 3.5 | 0.00x |
| conjunctival epithelial cell | 0 | 0.0 | 2686 | 3.3 | 0.00x |
| endothelial cell | 0 | 0.0 | 2342 | 2.9 | 0.00x |
| macrophage | 2 | 0.0 | 1444 | 1.8 | 0.01x |
| endothelial cell of lymphatic vessel | 0 | 0.0 | 824 | 1.0 | 0.00x |
| blood vessel endothelial cell | 0 | 0.0 | 714 | 0.9 | 0.00x |
| lymphocyte | 13 | 0.1 | 468 | 0.6 | 0.17x |
| pericyte | 13 | 0.1 | 402 | 0.5 | 0.19x |
| transit amplifying cell | 0 | 0.0 | 332 | 0.4 | 0.00x |
| goblet cell | 0 | 0.0 | 279 | 0.3 | 0.00x |
| leukocyte | 0 | 0.0 | 82 | 0.1 | 0.00x |
| mast cell | 0 | 0.0 | 21 | 0.0 | 0.00x |

### A.5. cell_type_raw (DEG enrichment) x leiden cluster — full table

```
leiden             0     1    10    11    12    13    14    15    16   17   18   19     2   20   21   22   23   24     3     4     5     6     7     8     9
cell_type_raw                                                                                                                                               
LPC                0     0     0  2654     0     0  1526     0  1256    0    0  659     0    0  385  306  232    0     0  5947  5434     0     0     0  3125
LSC            13584     0     0     0     0     0     0     0     0  826    0    0     0  486    0    0    0    0     0     0     0     0     0     0     0
PMC_basal          0  8460     0     0     0     0     0     0     0    0  724    0  8299    0    0    0    0  158     0     0     0  3976  3715     0     0
TDC                0     0  3018     0     0     0     0     0     0    0    0    0     0    0    0    0    0    0  7911     0     0     0     0  3543     0
conjunctival       0     0     0     0  1652     0     0     0     0    0    0    0     0    0    0    0    0    0     0     0     0     0     0     0     0
melanocyte         0     0     0     0     0  1641     0  1300     0    0    0    0     0    0    0    0    0    0     0     0     0     0     0     0     0
```

## B. Marker expression (counts layer; %exp = count>0, mean = mean log1p(counts))

Groups: **cluster0** (13,584) | **basal** = cell_type_raw PMC_basal (25,332) | **LSC_other** = cell_type_raw LSC excl. cluster 0 (clusters 17+20, 1,312) | **diff** = cell_type_raw TDC (14,472) | **NON_LAYER** (4,593).

NOTE: markers not found in 13,250 retained genes: ['BIRC5', 'MKI67', 'PLK1', 'RRM2', 'SOX17', 'TK1', 'TOP2A', 'TSPAN7']

### B.1. Epithelial markers

| gene | group | %exp | mean log1p |
|---|---|---|---|
| KRT14 | cluster0 | 0.18 | 0.8311 |
| KRT14 | basal | 18.75 | 1.5651 |
| KRT14 | LSC_other | 3.73 | 1.0121 |
| KRT14 | diff | 2.24 | 0.8556 |
| KRT15 | cluster0 | 0.77 | 0.8229 |
| KRT15 | basal | 13.52 | 2.0518 |
| KRT15 | LSC_other | 7.62 | 1.0038 |
| KRT15 | diff | 5.9 | 1.0097 |
| KRT19 | cluster0 | 3.19 | 0.785 |
| KRT19 | basal | 12.72 | 1.5314 |
| KRT19 | LSC_other | 16.54 | 0.8474 |
| KRT19 | diff | 16.95 | 1.4001 |
| KRT5 | cluster0 | 1.32 | 0.8136 |
| KRT5 | basal | 82.66 | 2.2185 |
| KRT5 | LSC_other | 12.65 | 0.9505 |
| KRT5 | diff | 80.44 | 1.9227 |
| KRT12 | cluster0 | 1.78 | 0.7863 |
| KRT12 | basal | 76.79 | 2.7714 |
| KRT12 | LSC_other | 9.98 | 0.9792 |
| KRT12 | diff | 82.55 | 2.3708 |
| KRT3 | cluster0 | 0.32 | 0.7574 |
| KRT3 | basal | 57.59 | 1.7687 |
| KRT3 | LSC_other | 3.43 | 0.9439 |
| KRT3 | diff | 71.79 | 1.9691 |
| COL17A1 | cluster0 | 0.53 | 0.7903 |
| COL17A1 | basal | 76.71 | 1.8302 |
| COL17A1 | LSC_other | 3.96 | 1.0483 |
| COL17A1 | diff | 41.87 | 1.149 |
| ITGB1 | cluster0 | 69.37 | 1.0881 |
| ITGB1 | basal | 57.85 | 1.0968 |
| ITGB1 | LSC_other | 44.28 | 1.0042 |
| ITGB1 | diff | 42.03 | 0.9407 |

### B.2. Cluster 0 top DEGs (CHRM3, DGKI, CARMN, MYH11)

| gene | group | %exp | mean log1p |
|---|---|---|---|
| CHRM3 | cluster0 | 99.99 | 4.0608 |
| CHRM3 | basal | 8.17 | 0.9295 |
| CHRM3 | LSC_other | 12.8 | 0.8138 |
| CHRM3 | diff | 2.54 | 0.7743 |
| CHRM3 | NON_LAYER | 17.35 | 1.2016 |
| DGKI | cluster0 | 99.93 | 3.7436 |
| DGKI | basal | 7.49 | 0.9415 |
| DGKI | LSC_other | 9.53 | 0.7928 |
| DGKI | diff | 15.45 | 1.1088 |
| DGKI | NON_LAYER | 17.4 | 1.1542 |
| CARMN | cluster0 | 99.85 | 3.1262 |
| CARMN | basal | 3.13 | 0.8768 |
| CARMN | LSC_other | 5.11 | 0.7993 |
| CARMN | diff | 0.37 | 0.7157 |
| CARMN | NON_LAYER | 7.53 | 1.2001 |
| MYH11 | cluster0 | 99.94 | 3.1804 |
| MYH11 | basal | 14.29 | 0.9439 |
| MYH11 | LSC_other | 11.74 | 0.8546 |
| MYH11 | diff | 4.11 | 0.7312 |
| MYH11 | NON_LAYER | 19.62 | 1.0696 |

### B.3. Stromal / TM markers

| gene | group | %exp | mean log1p |
|---|---|---|---|
| DCN | cluster0 | 22.39 | 0.8295 |
| DCN | basal | 12.38 | 1.0153 |
| DCN | LSC_other | 8.92 | 0.9197 |
| DCN | diff | 4.42 | 0.7619 |
| COL1A1 | cluster0 | 3.61 | 0.7354 |
| COL1A1 | basal | 2.62 | 0.7479 |
| COL1A1 | LSC_other | 1.37 | 0.7607 |
| COL1A1 | diff | 1.46 | 0.7221 |
| COL3A1 | cluster0 | 11.44 | 0.8087 |
| COL3A1 | basal | 2.44 | 0.8232 |
| COL3A1 | LSC_other | 1.75 | 0.9085 |
| COL3A1 | diff | 0.17 | 0.6931 |
| MYOC | cluster0 | 95.77 | 1.9394 |
| MYOC | basal | 9.52 | 0.9606 |
| MYOC | LSC_other | 10.37 | 0.9119 |
| MYOC | diff | 3.29 | 0.7383 |
| FN1 | cluster0 | 13.09 | 0.8408 |
| FN1 | basal | 4.86 | 0.7489 |
| FN1 | LSC_other | 15.55 | 1.0606 |
| FN1 | diff | 3.33 | 0.7204 |
| VIM | cluster0 | 73.23 | 1.2369 |
| VIM | basal | 16.67 | 1.2304 |
| VIM | LSC_other | 49.39 | 1.0065 |
| VIM | diff | 1.78 | 0.7266 |
| ACTA2 | cluster0 | 98.59 | 2.1429 |
| ACTA2 | basal | 9.68 | 0.8948 |
| ACTA2 | LSC_other | 8.23 | 0.8926 |
| ACTA2 | diff | 2.67 | 0.7845 |

### B.4. LSC panel (score_LSC genes: ABCG2, ITGA9, TSPAN7, SOX17)

| gene | group | %exp | mean log1p |
|---|---|---|---|
| ABCG2 | cluster0 | 2.35 | 0.7516 |
| ABCG2 | basal | 1.83 | 0.7994 |
| ABCG2 | LSC_other | 1.3 | 0.8055 |
| ABCG2 | diff | 0.8 | 0.7071 |
| ITGA9 | cluster0 | 12.58 | 0.8137 |
| ITGA9 | basal | 14.34 | 0.853 |
| ITGA9 | LSC_other | 57.85 | 1.7284 |
| ITGA9 | diff | 5.6 | 0.8028 |
| TSPAN7 | cluster0 | nan | nan |
| TSPAN7 | basal | nan | nan |
| TSPAN7 | LSC_other | nan | nan |
| TSPAN7 | diff | nan | nan |
| SOX17 | cluster0 | nan | nan |
| SOX17 | basal | nan | nan |
| SOX17 | LSC_other | nan | nan |
| SOX17 | diff | nan | nan |

## C. TAC / cycling-program investigation

NOTE: all 6 TAC proliferation genes (MKI67, BIRC5, RRM2, TOP2A, PLK1, TK1) were REMOVED from the QC gene set (13,250 genes) by the locked min_cells=3% gene filter (rare-expression markers; the Stage 1 `scoring_note` pre-announced exactly this). They therefore do not exist in `layers['counts']`. %exp below is computed from the raw pre-QC partition matrices (59,413 genes) and aligned to the 80,817 QC-passing cells via unique `observation_joinid`.

### C.1. TAC 6-gene %exp per Leiden cluster (raw pre-QC counts>0, aligned)

| cluster | n | MKI67 | BIRC5 | RRM2 | TOP2A | PLK1 | TK1 | any_of_6 |
|---|---|---|---|---|---|---|---|---|
| 0 | 13584 | 0.64 | 0.26 | 1.02 | 0.54 | 0.46 | 0.35 | 3.22 |
| 1 | 8460 | 0.2 | 0.83 | 4.46 | 0.7 | 1.42 | 0.51 | 7.49 |
| 2 | 8299 | 0.74 | 0.94 | 6.28 | 2.08 | 1.8 | 1.57 | 10.15 |
| 3 | 7911 | 0.03 | 1.05 | 3.29 | 0.48 | 1.69 | 0.9 | 7.21 |
| 4 | 5947 | 0.05 | 0.24 | 1.38 | 0.52 | 1.01 | 0.24 | 3.36 |
| 5 | 5434 | 0.02 | 0.4 | 0.98 | 0.22 | 0.98 | 0.57 | 3.13 |
| 6 | 3976 | 0.15 | 0.15 | 0.6 | 0.43 | 0.73 | 0.53 | 2.54 |
| 7 | 3715 | 0.7 | 0.92 | 2.02 | 1.86 | 1.48 | 0.75 | 5.87 |
| 8 | 3543 | 0.06 | 2.37 | 3.3 | 0.23 | 1.89 | 1.27 | 8.72 |
| 9 | 3125 | 0.03 | 0.35 | 0.7 | 0.64 | 0.45 | 1.25 | 3.30 |
| 10 | 3018 | 0.1 | 0.6 | 4.14 | 0.73 | 1.33 | 0.96 | 7.12 |
| 11 | 2654 | 0.11 | 0.3 | 1.92 | 0.3 | 0.64 | 0.57 | 3.81 |
| 12 | 1652 | 0.0 | 0.48 | 0.91 | 0.36 | 1.21 | 4.48 | 7.20 |
| 13 | 1641 | 0.0 | 0.0 | 0.55 | 0.43 | 0.55 | 0.12 | 1.65 |
| 14 | 1526 | 0.98 | 0.46 | 1.7 | 1.44 | 0.66 | 1.51 | 5.31 |
| 15 | 1300 | 0.23 | 0.15 | 1.38 | 0.77 | 0.77 | 0.15 | 3.38 |
| 16 | 1256 | 0.0 | 0.4 | 1.43 | 0.48 | 1.43 | 0.08 | 3.66 |
| 17 | 826 | 0.0 | 0.12 | 0.73 | 0.48 | 0.48 | 0.48 | 2.30 |
| 18 | 724 | 0.83 | 1.1 | 3.04 | 1.8 | 1.8 | 1.38 | 6.77 |
| 19 | 659 | 0.0 | 0.15 | 0.61 | 0.3 | 0.76 | 0.15 | 1.97 |
| 20 | 486 | 0.21 | 0.62 | 1.23 | 0.62 | 0.21 | 0.41 | 2.47 |
| 21 | 385 | 0.26 | 0.26 | 1.3 | 0.52 | 1.3 | 0.0 | 3.64 |
| 22 | 306 | 0.0 | 0.65 | 2.94 | 0.65 | 1.96 | 0.65 | 6.54 |
| 23 | 232 | 0.0 | 0.0 | 2.59 | 0.43 | 1.72 | 2.16 | 6.90 |
| 24 | 158 | 0.0 | 1.27 | 1.9 | 0.0 | 0.63 | 1.27 | 5.06 |

### C.2. Co-expression of >=2 TAC genes (any pair, QC cells)
- QC cells (80,817): expressing >=2 of 6 TAC genes: **311** (0.385%); expressing >=1: 4369 (5.406%).
- Context, all 89,794 pre-QC cells: >=2 TAC genes: 402 (0.448%).

### C.3. score_TAC distribution (obs; scored pre-QC on full gene set)

| quantile | value |
|---|---|
| 0% | -0.2150 |
| 5% | -0.0350 |
| 25% | -0.0200 |
| 50% | -0.0100 |
| 75% | -0.0050 |
| 90% | 0.0000 |
| 95% | 0.1317 |
| 99% | 0.1667 |
| 100% | 6.3950 |

### C.4. score_TAC > gt95pct (0.132): 3908 cells (4.836%) — cluster composition

| cluster | n | % of high-score cells | cell_type_raw |
|---|---|---|---|
| 0 | 388 | 9.9 | LSC(388) |
| 1 | 599 | 15.3 | PMC_basal(599) |
| 2 | 761 | 19.5 | PMC_basal(761) |
| 3 | 524 | 13.4 | TDC(524) |
| 4 | 156 | 4.0 | LPC(156) |
| 5 | 152 | 3.9 | LPC(152) |
| 6 | 92 | 2.4 | PMC_basal(92) |
| 7 | 191 | 4.9 | PMC_basal(191) |
| 8 | 286 | 7.3 | TDC(286) |
| 9 | 90 | 2.3 | LPC(90) |
| 10 | 201 | 5.1 | TDC(201) |
| 11 | 80 | 2.0 | LPC(80) |
| 12 | 87 | 2.2 | conjunctival(87) |
| 13 | 21 | 0.5 | melanocyte(21) |
| 14 | 65 | 1.7 | LPC(65) |
| 15 | 37 | 0.9 | melanocyte(37) |
| 16 | 40 | 1.0 | LPC(40) |
| 17 | 19 | 0.5 | LSC(19) |
| 18 | 48 | 1.2 | PMC_basal(48) |
| 19 | 11 | 0.3 | LPC(11) |
| 20 | 11 | 0.3 | LSC(11) |
| 21 | 14 | 0.4 | LPC(14) |
| 22 | 16 | 0.4 | LPC(16) |
| 23 | 11 | 0.3 | LPC(11) |
| 24 | 8 | 0.2 | PMC_basal(8) |

### C.4. score_TAC > gt99pct (0.167): 635 cells (0.786%) — cluster composition

| cluster | n | % of high-score cells | cell_type_raw |
|---|---|---|---|
| 0 | 27 | 4.3 | LSC(27) |
| 1 | 103 | 16.2 | PMC_basal(103) |
| 2 | 245 | 38.6 | PMC_basal(245) |
| 3 | 50 | 7.9 | TDC(50) |
| 4 | 9 | 1.4 | LPC(9) |
| 5 | 7 | 1.1 | LPC(7) |
| 6 | 5 | 0.8 | PMC_basal(5) |
| 7 | 52 | 8.2 | PMC_basal(52) |
| 8 | 27 | 4.3 | TDC(27) |
| 9 | 6 | 0.9 | LPC(6) |
| 10 | 40 | 6.3 | TDC(40) |
| 11 | 6 | 0.9 | LPC(6) |
| 12 | 10 | 1.6 | conjunctival(10) |
| 14 | 18 | 2.8 | LPC(18) |
| 15 | 2 | 0.3 | melanocyte(2) |
| 16 | 3 | 0.5 | LPC(3) |
| 17 | 2 | 0.3 | LSC(2) |
| 18 | 17 | 2.7 | PMC_basal(17) |
| 20 | 2 | 0.3 | LSC(2) |
| 21 | 1 | 0.2 | LPC(1) |
| 22 | 2 | 0.3 | LPC(2) |
| 24 | 1 | 0.2 | PMC_basal(1) |

## Evidence chain

### E1. Cluster 0 identity: ciliary muscle (smooth muscle), not corneal epithelium

| Evidence line | Result |
|---|---|
| CELLxGENE `author_cell_type` (van Zyl's own) | 99.2% (13,477/13,584) = **Ciliary_Muscle**; cluster 0 contains 13,477 of the dataset's 13,592 ciliary muscle cells (**99.2% of all ciliary muscle cells**) |
| CELLxGENE curated `cell_type` (independent) | 99.2% (13,477) = **ciliary muscle cell** — two independent annotation systems agree |
| Partition | **100% TM wedge** (0/13,584 from central cornea) — consistent with ciliary body location |
| Smooth-muscle program | MYH11 99.94%, CHRM3 99.99%, ACTA2 98.59%, MYOC 95.77%, CARMN 99.85%, DGKI 99.93% (cluster 0 mean log1p 1.9–4.1, vs <15% in all epithelial reference groups) |
| Epithelial program | KRT5 1.32% (basal 82.66%), KRT12 1.78% (diff 82.55%), KRT14 0.18% (basal 18.75%), KRT3 0.32% (diff 71.79%), COL17A1 0.53% (basal 76.71%), KRT15 0.77% — **effectively zero keratin expression**; ITGB1 69.4% is ubiquitous (57.9% in basal) and lineage-neutral |

**How cluster 0 became "LSC":** DEG-enrichment assigned LSC with score 0.095 — not because of LSC markers, but because all 8 epithelial enrichments were ≈0 (non-epithelial sets also below the 0.3 gate), and the locked logic's else-branch returns `max(epithelial_types, ...)` → first key `"LSC"`. Cluster 0 is a forced-choice artifact of the assignment rule on a genuinely non-epithelial cluster. This is a known methodological property of the locked pipeline (same rule produced e.g. cluster 20 → LSC with all enrichments 0.0).

### E2. TAC absence: pipeline-structure artifact + genuinely rare cycling cells

| Evidence line | Result |
|---|---|
| Gene-set availability | All 6 TAC genes (MKI67, BIRC5, RRM2, TOP2A, PLK1, TK1) were **removed from the 13,250-gene QC set** by the locked min_cells=3% filter (present in raw 59,413-gene matrices; `scoring_note` pre-announced this) |
| Consequence | `rank_genes_groups` ran without any TAC marker → **TAC enrichment = 0 for every cluster by construction** → zero clusters could ever be assigned TAC. The empty TAC class in the 4-class output is fully explained. |
| Cycling cells are rare | QC cells with ≥2 of 6 TAC genes: **311 (0.385%)**; ≥1: 4,369 (5.41%); pre-QC (89,794) ≥2: 402 (0.448%) |
| No dedicated cycling cluster | Per-cluster any_of_6 %exp: max 10.15% (cl 2), then 8.72% (cl 8), 7.49% (cl 1), 7.21% (cl 3); cluster 0 = 3.22% (mid-low). MKI67 alone ≤0.98% in every cluster (cl 14). |
| High score_TAC cells spread out | >95th pct (0.132): 3,908 cells across all 25 clusters; >99th pct (0.167): 635 cells, mildly concentrated in PMC_basal clusters 1+2 (54.8%) — basal is the physiologic proliferative compartment |
| Bottom line | Cycling cells exist but are too rare (≈0.4%) to drive any 0.6-resolution cluster; TAC class would stay empty even with genes present unless the resolution/gene-filter is changed |

### E3. Side effects for the decision (no change made — for the science team)

- Excluding cluster 0: 80,817 → 67,233 cells; is_epithelial 76,224 → 62,640; cell_type_raw LSC 14,896 → 1,312.
- Cluster 0 contains ≈107 non-ciliary-muscle cells (76 TM_Fibro, 13 Lymphocyte, 13 Pericyte, 2 Uveal_Fibro, 2 Macrophage, 1 Schwann = 0.8%) — all non-epithelial; no epithelial cells would be lost by exclusion.

## Recommendation (tendency, not final decision)

**Cluster 0: EXCLUDE from the G1 epithelial (4-class) analysis** — confidence **high**.

Rationale: five independent lines of evidence converge (author annotation 99.2% Ciliary_Muscle; curated cell_type 99.2% ciliary muscle; smooth-muscle program at ~100% expression; keratin program at ~0%; 100% TM-wedge origin). Cluster 0 is the dataset's ciliary-muscle population, and its LSC label is a forced-choice artifact (enrichment 0.095). If non-epithelial cells should be retained for downstream analyses, **REASSIGN cluster 0 to NON_LAYER** is the equivalent alternative; EXCLUDE vs REASSIGN is a scope question for the team (G1 epithelial-only → EXCLUDE; whole-atlas analyses → REASSIGN).

**TAC class**: no evidence supports a hidden cycling subpopulation at res 0.6. If G1 requires a TAC class, the pipeline itself needs a change (e.g., DEG on pre-filter gene set or relaxed min_cells) — outside this task's constraints; flagged, not implemented.

**Caveat**: cluster 0 = 16.8% of all cells; its removal materially changes composition — the science team should confirm the decision explicitly before G1.
