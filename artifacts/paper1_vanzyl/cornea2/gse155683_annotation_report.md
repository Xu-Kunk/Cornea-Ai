# GSE155683 — Annotation report (protocol_15d, mirrors 15c)

- Cells in: 21343; after QC: 19208
- Genes in: 33538; after QC: 11592
- Leiden clusters (res=0.6): 18

## cell_class_4 counts

```
cell_class_4
NON_LAYER         6299
differentiated    6247
LSC               3511
basal             3151
```

## Per-donor cell_class_4

```
cell_class_4   LSC  NON_LAYER  basal  differentiated
donor_id                                            
donor_1        453       1175    403             602
donor_2        621        627    291             876
donor_3        282       1611    162             114
donor_4       2155       2886   2295            4655
```

## Cluster assignment (DEG enrichment, wilcoxon top-200)

```
          LSC    LPC  TAC  PMC_basal    TDC  Melanocyte  Conjunctival  Immune  n_cells              assigned top_deg_1 top_deg_2 top_deg_3 top_deg_4 top_deg_5
leiden                                                                                                                                                        
0       0.000  0.191  0.0      0.042  0.126       0.000         0.199   0.000     2576                   LPC   S100A11     KRT13      AQP3     KRT19      SLPI
1       0.000  0.000  0.0      0.309  0.488       0.000         0.000   0.000     2542                   TDC       TKT     KRT12     DAPL1     TGFBI     MGARP
2       0.000  0.300  0.0      0.425  0.154       0.000         0.000   0.000     2393             PMC_basal    CXCL14     KRT15   COL17A1       CLU      MT1X
3       0.087  0.191  0.0      0.249  0.000       0.000         0.000   0.000     2278             PMC_basal       DCN    ITGBL1     HTRA1       LUM      KERA
4       0.000  0.000  0.0      0.171  0.545       0.000         0.000   0.000     2112                   TDC     ADIRF      MAL2     LYPD3      DSC2      EMP1
5       0.000  0.000  0.0      0.020  0.238       0.000         0.152   0.000     1593                   TDC     KRT24    LGALS3     LYPD2      SCEL   HRASLS2
6       0.000  0.116  0.0      0.000  0.041       0.000         0.567   0.000     1373          conjunctival      AQP5        F3      LCN2    S100A9    CXCL17
7       0.000  0.246  0.0      0.041  0.000       0.000         0.000   0.000      923                   LPC     FBLN1       C1S       MGP       CFH       C1R
8       0.000  0.249  0.0      0.220  0.044       0.000         0.000   0.000      747                   LPC    S100A2     KRT14     RPLP1     RPS27     RPS19
9       0.583  0.299  0.0      0.049  0.000       0.000         0.000   0.139      693                   LSC   SPARCL1      CD74    PECAM1     CLDN5  HLA-DRB1
10      0.000  0.122  0.0      0.718  0.193       0.000         0.000   0.000      664             PMC_basal   COL17A1    CXCL14     GAPDH      BCAM       DST
11      0.108  0.101  0.0      0.057  0.000       0.000         0.000   0.000      340                   LSC     TAGLN    IGFBP7      MYL9     NR2F2     CALD1
12      0.000  0.231  0.0      0.118  0.000       0.000         0.000   0.000      208                   LPC    COL1A1    COL3A1    COL1A2     SPARC       GSN
13      0.000  0.151  0.0      0.000  0.000       0.000         0.000   0.329      202  immune_contamination   HLA-DRA      SRGN      CD74  HLA-DPA1  HLA-DRB1
14      0.000  0.090  0.0      0.000  0.000       0.000         0.000   0.000      188                   LPC     RPS27     RPS29      SRGN    RPS27A     HSPA8
15      0.119  0.062  0.0      0.030  0.000       0.000         0.000   0.000      151                   LSC     MMRN1     CCL21      TFPI     GNG11   PPFIBP1
16      0.000  0.000  0.0      0.000  0.000       0.399         0.000   0.000      131            melanocyte       DCT      PMEL      APOE     EDNRB      QPCT
17      0.058  0.000  0.0      0.073  0.000       0.000         0.000   0.000       94             PMC_basal       VIM     STMN1     GPM6B    HSPA1A     FXYD1
```

## Cross-check

```json
{
  "dataset": "GSE155683",
  "note": "No per-cell metadata exists on GEO for this dataset."
}
```
