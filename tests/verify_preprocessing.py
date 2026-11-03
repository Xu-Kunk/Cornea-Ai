"""
Corneal-FM Preprocessing Verification Report
the journal  — Research Article
Generated: 2026-07-06
"""
import os, sys, json, hashlib
from pathlib import Path
from datetime import datetime
import scanpy as sc
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent  # portable: derive from script location
DATA = ROOT / "data"
SCRIPTS = ROOT / "scripts"
DOCS = ROOT / "docs"
DOCS.mkdir(exist_ok=True)

report_lines = []
def rpt(s=""):
    report_lines.append(s)
    print(s)

# ============================================================
# HEADER
# ============================================================
rpt(f"# Corneal-FM Preprocessing Report")
rpt(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
rpt(f"Release package")
rpt()

# ============================================================
# 1. ENVIRONMENT
# ============================================================
rpt("## 1. Environment")
rpt()
import sys, torch, transformers, scanpy as sc2, anndata
rpt(f"| Package | Version |")
rpt(f"|---------|---------|")
rpt(f"| Python | {sys.version.split()[0]} |")
rpt(f"| PyTorch | {torch.__version__} |")
rpt(f"| transformers | {transformers.__version__} |")
rpt(f"| scanpy | {sc2.__version__} |")
rpt(f"| anndata | {anndata.__version__} |")
try:
    import geneformer
    rpt(f"| geneformer | {geneformer.__version__} |")
except: rpt(f"| geneformer | installed (no __version__) |")
rpt(f"| Platform | {sys.platform} |")
rpt()

# ============================================================
# 2. DATA PROVENANCE
# ============================================================
rpt("## 2. Data Provenance")
rpt()
rpt(f"| Dataset | Source | Accession |")
rpt(f"|---------|--------|-----------|")
rpt(f"| Human Ocular Surface Atlas | CELLxGENE Discover | 0f7d022a-46c7-4e64-be4c-e34adbb78089 |")
rpt(f"| Mouse DED | DED-seq Project / GSE182582 | GSE182582 (lacrimal gland excision) |")
rpt(f"| Ensembl Orthologs | BioMart REST API | Release 113 |")
rpt(f"| scGPT Vocab | bowang-lab/scGPT | default_gene_vocab.json |")
rpt()

# Source file integrity
src = DATA / "30127f02-d0b9-4b56-bf4c-e5b77e4989bd.h5ad"
if src.exists():
    md5 = hashlib.md5(src.read_bytes()).hexdigest()
    sz = src.stat().st_size
    rpt(f"**Source integrity**: {src.name}")
    rpt(f"- Size: {sz:,} bytes ({sz/1e9:.2f} GB)")
    rpt(f"- MD5: {md5}")
    rpt(f"- Expected: 10,293,731,327 bytes ✅" if sz == 10293731327 else f"⚠️ Size mismatch!")
rpt()

# ============================================================
# 3. SUBSAMPLING
# ============================================================
rpt("## 3. Subsampling (120k → 100k train + 20k test)")
rpt()
train_f = DATA / "corneal_100k_subsample.h5ad"
test_f = DATA / "corneal_20k_test.h5ad"

checks = []
for label, f in [("Train", train_f), ("Test", test_f)]:
    if not f.exists():
        rpt(f"⚠️ {label} file MISSING: {f}")
        checks.append(False)
        continue
    a = sc.read_h5ad(f, backed='r')
    sz = f.stat().st_size
    n_ct = a.obs['cell_type'].nunique()
    rare_ct = a.obs['cell_type'].value_counts().iloc[-1]
    rpt(f"**{label}**: {a.n_obs:,} cells × {a.n_vars:,} genes ({sz/1e9:.2f} GB)")
    rpt(f"- Cell types: {n_ct}/20, rarest: {rare_ct}")
    rpt(f"- Has raw: {a.raw is not None}")
    checks.append(n_ct == 20 and a.n_obs > 0)

rpt(f"- All checks: {'✅ PASS' if all(checks) else '❌ FAIL'}")

# Check no overlap
if train_f.exists() and test_f.exists():
    a_tr = sc.read_h5ad(train_f, backed='r')
    a_te = sc.read_h5ad(test_f, backed='r')
    overlap = len(set(a_tr.obs_names) & set(a_te.obs_names))
    rpt(f"- Train/test overlap: {overlap} cells {'✅' if overlap == 0 else '❌ OVERLAP!'}")
rpt()

# ============================================================
# 4. GENE FORMER TOKENIZATION
# ============================================================
rpt("## 4. Geneformer Tokenization")
rpt()
gf_train = DATA / "gf_tokenized_train"
gf_test = DATA / "gf_tokenized_test"

for label, p in [("Train", gf_train), ("Test", gf_test)]:
    if not p.exists():
        rpt(f"⚠️ {label}: NOT FOUND at {p}")
        continue
    try:
        from datasets import load_from_disk
        ds = load_from_disk(str(p))
        n = len(ds)
        cols = ds.column_names
        rpt(f"**{label}**: {n:,} cells")
        rpt(f"- Columns: {cols}")
        if 'length' in cols:
            rpt(f"- Avg tokens/cell: {np.mean(ds['length']):.0f}")
            rpt(f"- Max tokens/cell: {np.max(ds['length'])}")
    except Exception as e:
        rpt(f"⚠️ {label}: Failed to load — {e}")
rpt()

# ============================================================
# 5. scGPT PREPARATION
# ============================================================
rpt("## 5. scGPT Data Preparation")
rpt()
scgpt_f = DATA / "corneal_scgpt_ready.h5ad"
vocab_f = DATA / "scgpt_gene_vocab.json"
if scgpt_f.exists():
    a = sc.read_h5ad(scgpt_f, backed='r')
    rpt(f"**Output**: {a.n_obs:,} cells × {a.n_vars:,} genes")
    rpt(f"- Size: {scgpt_f.stat().st_size/1e9:.2f} GB")
    rpt(f"- Has raw: {a.raw is not None}")
if vocab_f.exists():
    with open(vocab_f) as f:
        v = json.load(f)
    rpt(f"**Vocab**: {len(v):,} tokens")
rpt()

# ============================================================
# 6. MOUSE DED
# ============================================================
rpt("## 6. Mouse DED Data")
rpt()
mouse_f = DATA / "mouse_ded.h5ad"
if mouse_f.exists():
    a = sc.read_h5ad(mouse_f)
    rpt(f"**Mouse DED**: {a.n_obs:,} cells × {a.n_vars:,} genes")
    rpt(f"- Size: {mouse_f.stat().st_size/1e6:.1f} MB")
    rpt(f"- Condition: {dict(a.obs['condition'].value_counts())}")
    if 'celltype_orig' in a.obs:
        rpt(f"- Cell types: {a.obs['celltype_orig'].nunique()}")
    rpt(f"- Has raw counts: {'X' in str(type(a.X))}")
else:
    rpt("⚠️ NOT FOUND — cross-species E3 will need manual download")
rpt()

# ============================================================
# 7. ORTHOLOGS
# ============================================================
rpt("## 7. Ortholog Mapping")
rpt()
ortho_f = DATA / "mouse_human_orthologs.csv"
if ortho_f.exists():
    df = pd.read_csv(ortho_f)
    rpt(f"**Mouse→Human 1:1 orthologs**: {len(df):,} pairs")
    rpt(f"- Columns: {list(df.columns)}")
    rpt(f"- Unique mouse genes: {df['mouse_gene_symbol'].nunique():,}")
    rpt(f"- Unique human genes: {df['human_gene_symbol'].nunique():,}")
    rpt(f"- Duplicate mappings: {df.duplicated(subset=['mouse_gene_symbol']).sum()}")
    rpt(f"- Size: {ortho_f.stat().st_size/1e6:.2f} MB")
    rpt(f"- Source: Ensembl Release 113 (BioMart)")
else:
    rpt("⚠️ NOT FOUND")
rpt()

# ============================================================
# 8. DIRECTORY SUMMARY
# ============================================================
rpt("## 8. Output Directory")
rpt()
rpt("```")
for f in sorted(DATA.glob("*")):
    if f.is_file():
        sz = f.stat().st_size
        rpt(f"  {f.name:<45} {sz/1e6:>8.1f} MB" if sz < 1e9 else f"  {f.name:<45} {sz/1e9:>8.2f} GB")
    elif f.is_dir():
        rpt(f"  {f.name}/ (dir)")
rpt("```")
rpt()

# ============================================================
# 9. SCRIPTS INVENTORY
# ============================================================
rpt("## 9. Scripts")
rpt()
for f in sorted(SCRIPTS.glob("*.py")):
    rpt(f"- `scripts/{f.name}`")
rpt()

# ============================================================
# WRITE REPORT
# ============================================================
report_path = DOCS / "preprocessing_report.md"
with open(report_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(report_lines))
print(f"\nReport saved: {report_path}")
