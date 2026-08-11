"""Verify CELLxGENE Ocular Surface Atlas h5ad file integrity."""
import os
from pathlib import Path

import os
from pathlib import Path

import h5py
import anndata as ad
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("CORNEAL_FM_DATA", str(PROJECT_ROOT / "data")))
FPATH = DATA_DIR / "30127f02-d0b9-4b56-bf4c-e5b77e4989bd.h5ad"

print("=" * 60)
print("CELLxGENE Ocular Surface Atlas — Verification")
print("=" * 60)

# 1. h5py structural check
print("\n[1] h5py structural integrity...")
f = h5py.File(FPATH, 'r')
print(f"  X group keys: {list(f['X'].keys())}")
print(f"  obs columns: {len(list(f['obs'].dtype.names))}")
print(f"  var columns: {len(list(f['var'].dtype.names))}")
f.close()
print("  OK — file not corrupted")

# 2. anndata load
print("\n[2] Loading with anndata (backed mode)...")
adata = ad.read_h5ad(FPATH, backed='r')
print(f"  Cells: {adata.n_obs:,}")
print(f"  Genes: {adata.n_vars:,}")
print(f"  Shape: {adata.shape}")

# 3. Key metadata
print("\n[3] Metadata summary:")
print(f"  Cell types: {adata.obs['cell_type'].nunique()}")
print(f"  Tissues: {adata.obs['tissue'].nunique()}")
print(f"  Diseases: {adata.obs['disease'].value_counts().to_dict()}")
print(f"  Studies: {adata.obs['study'].nunique()}")
print(f"  Donors: {adata.obs['donor_id'].nunique()}")

# 4. Cell type distribution
print("\n[4] Top 15 cell types:")
ct = adata.obs['cell_type'].value_counts()
for i, (name, count) in enumerate(ct.head(15).items()):
    pct = count / adata.n_obs * 100
    print(f"  {i+1:2}. {name:<40} {count:>8,} ({pct:5.1f}%)")

# 5. Tissue distribution
print("\n[5] Tissue distribution:")
for tissue, count in adata.obs['tissue'].value_counts().items():
    pct = count / adata.n_obs * 100
    print(f"  {tissue:<40} {count:>8,} ({pct:5.1f}%)")

# 6. Dataset version check
print(f"\n[6] Data source:")
if 'citation' in adata.uns:
    print(f"  Citation: {adata.uns['citation']}")
print(f"  Total genes per cell (mean): {adata.obs['nFeature_RNA'].mean():.0f}")

print("\n" + "=" * 60)
print("VERIFICATION COMPLETE — File ready for use")
print("=" * 60)
