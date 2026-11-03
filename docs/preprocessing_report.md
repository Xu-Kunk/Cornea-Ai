# Corneal-FM Preprocessing Report
Generated: 2026-07-06 13:46:07
Target: Patterns (IF 10.8), Research Article

## 1. Environment

| Package | Version |
|---------|---------|
| Python | 3.12.10 |
| PyTorch | 2.12.1+cpu |
| transformers | 4.45.2 |
| scanpy | 1.12.2 |
| anndata | 0.12.19 |
| geneformer | installed (no __version__) |
| Platform | win32 |

## 2. Data Provenance

| Dataset | Source | Accession |
|---------|--------|-----------|
| Human Ocular Surface Atlas | CELLxGENE Discover | 0f7d022a-46c7-4e64-be4c-e34adbb78089 |
| Mouse DED | DED-seq Project / GSE182582 | GSE182582 (lacrimal gland excision) |
| Ensembl Orthologs | BioMart REST API | Release 113 |
| scGPT Vocab | bowang-lab/scGPT | default_gene_vocab.json |

**Source integrity**: 30127f02-d0b9-4b56-bf4c-e5b77e4989bd.h5ad
- Size: 10,293,731,327 bytes (10.29 GB)
- MD5: 39f098367af95bc3e1ee92528ceb251e
- Expected: 10,293,731,327 bytes ✅

## 3. Subsampling (120k → 100k train + 20k test)

**Train**: 100,000 cells × 35,477 genes (4.28 GB)
- Cell types: 20/20, rarest: 28
- Has raw: True
**Test**: 20,000 cells × 35,477 genes (0.87 GB)
- Cell types: 20/20, rarest: 6
- Has raw: True
- All checks: ✅ PASS
- Train/test overlap: 0 cells ✅

## 4. Geneformer Tokenization

⚠️ Train: Failed to load — Directory D:\Codex\Ai\data\gf_tokenized_train is neither a `Dataset` directory nor a `DatasetDict` directory.
⚠️ Test: Failed to load — Directory D:\Codex\Ai\data\gf_tokenized_test is neither a `Dataset` directory nor a `DatasetDict` directory.

## 5. scGPT Data Preparation

**Output**: 815,769 cells × 4,251 genes
- Size: 5.52 GB
- Has raw: True
**Vocab**: 48,292 tokens

## 6. Mouse DED Data

**Mouse DED**: 1,000 cells × 17,802 genes
- Size: 5.9 MB
- Condition: {'control': np.int64(529), 'DED': np.int64(471)}
- Cell types: 11
- Has raw counts: False

## 7. Ortholog Mapping

**Mouse→Human 1:1 orthologs**: 17,066 pairs
- Columns: ['mouse_gene_symbol', 'mouse_ensembl_id', 'human_ensembl_id', 'human_gene_symbol']
- Unique mouse genes: 17,063
- Unique human genes: 17,052
- Duplicate mappings: 3
- Size: 0.83 MB
- Source: Ensembl Release 113 (BioMart)

## 8. Output Directory

```
  30127f02-d0b9-4b56-bf4c-e5b77e4989bd.h5ad        10.29 GB
  corneal_100k_subsample.h5ad                       4.28 GB
  corneal_20k_test.h5ad                            875.0 MB
  corneal_scgpt_ready.h5ad                          5.52 GB
  gf_tokenized_test/ (dir)
  gf_tokenized_train/ (dir)
  GSE182582_forGEO_controlVSded_countmatrix.csv.gz      4.1 MB
  mouse_ded.h5ad                                     5.9 MB
  mouse_human_orthologs.csv                          0.8 MB
  scgpt_gene_vocab.json                              0.8 MB
```

## 9. Scripts

- `scripts/download_cellxgene.py`
- `scripts/download_mouse_ded.py`
- `scripts/ensembl_orthologs.py`
- `scripts/prep_scgpt.py`
- `scripts/subsample.py`
- `scripts/subsample_v2.py`
- `scripts/subsample_v3.py`
- `scripts/tokenize_gf.py`
- `scripts/verify_gf.py`
- `scripts/verify_h5ad.py`
- `scripts/verify_pipeline.py`
