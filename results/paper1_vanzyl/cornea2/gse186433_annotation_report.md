# GSE186433 — Annotation report (protocol_15d, mirrors 15c)

- Cells in: 5964; after QC: 5367
- Genes in: 33525; after QC: 9125
- Leiden clusters (res=0.6): 8

## cell_class_4 counts

```
cell_class_4
differentiated    2400
LSC               1316
basal             1223
NON_LAYER          428
```

## Per-donor cell_class_4

```
cell_class_4  LSC  basal  differentiated  NON_LAYER
donor_id                                           
01             16     36             170        154
02            248    341             185         25
03              4     35              63         13
04             72    436             940         38
05            167     85              64         49
06            326     50             639        149
07+08         483    240             339          0
```

## Cluster assignment (DEG enrichment, wilcoxon top-200)

```
          LSC    LPC  TAC  PMC_basal    TDC  Melanocyte  Conjunctival  Immune  n_cells   assigned top_deg_1 top_deg_2 top_deg_3 top_deg_4 top_deg_5
leiden                                                                                                                                             
0       0.000  0.222  0.0      0.195  0.000         0.0           0.0     0.0     1316        LPC    S100A2     RPS3A     RPLP1    RPS27A     RPL10
1       0.000  0.000  0.0      0.000  0.402         0.0           0.0     0.0      798        TDC    LGALS3     LYPD2      HOPX     PPDPF    S100A4
2       0.000  0.000  0.0      0.172  0.660         0.0           0.0     0.0      779        TDC     KRT12      KRT3      ELF3     LYPD3       DSP
3       0.000  0.000  0.0      0.515  0.124         0.0           0.0     0.0      730  PMC_basal   COL17A1       DST     TGFBI   ALDH3A1    CRTAC1
4       0.000  0.057  0.0      0.047  0.480         0.0           0.0     0.0      496        TDC      SLPI     KRT6A      CSTB  SERPINB1      AQP3
5       0.000  0.088  0.0      0.399  0.040         0.0           0.0     0.0      493  PMC_basal     GAPDH   S100A10    IGFBP7      FTH1     RPL30
6       0.047  0.230  0.0      0.055  0.000         0.0           0.0     0.0      428        LPC      APOD       DCN     PTGDS      SAA1       VIM
7       0.000  0.000  0.0      0.059  0.439         0.0           0.0     0.0      327        TDC     GSTP1    S100A4     FABP5     ADIRF     PRDX5
```

## Cross-check

```json
{
  "dataset": "GSE186433",
  "note": "Layers is an anatomical-dissection label, not a cell-type annotation; reported descriptively only.",
  "per_layer_epithelial_fraction": {
    "Epi": 0.9203
  },
  "crosstab": {
    "LSC": {
      "Epi": 1316
    },
    "TAC": {
      "Epi": 0
    },
    "basal": {
      "Epi": 1223
    },
    "differentiated": {
      "Epi": 2400
    },
    "NON_LAYER": {
      "Epi": 428
    }
  }
}
```
